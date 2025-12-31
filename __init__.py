import bpy, gpu, subprocess, tempfile
from gpu_extras.batch import batch_for_shader
from pathlib import Path
import colorsys
import sys
import os
import urllib.request
import shutil
import ssl
import re

bl_info = {
    "name": "BB Waveform",
    "author": "Blender Bob & Claude.ai",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "Dopesheet Editor, Graph Editor > UI > BB Waveform",
    "description": "Display audio waveforms in timeline, dopesheet, and graph editors with FFmpeg auto-install. Compatible with Blender 4.2-5.0+",
    "category": "Animation",
}

# Constants
BASE_WAVEFORM_WIDTH = 4000
BASE_WAVEFORM_HEIGHT = 100
PIXELS_PER_FRAME_PER_LEVEL = 4
MAX_AUDIO_TRACKS = 16
COLOR_UPDATE_DELAY = 0.3  # seconds

# Minimal global state (will be cleaned up during refactor)
_handlers = []
_waveform_image = None
_waveform_coords = None
_rebuilding = False
_color_index = 0  # Track which color to assign next


def get_addon_name():
    """Get the addon name/package reliably"""
    return __name__


def get_sequences(seq_editor):
    """Get sequences from SequenceEditor - compatible with Blender 4.x and 5.0+
    Blender 5.0 changed the API structure for accessing sequences"""
    # Try different possible attribute names used across versions
    for attr in ['sequences_all', 'sequences', 'strips', 'all_sequences']:
        if hasattr(seq_editor, attr):
            return getattr(seq_editor, attr)
    
    # Fallback to empty list if attribute not found
    return []


def get_ffmpeg_storage_path():
    """Get persistent storage path for FFmpeg binaries (survives addon reinstall)"""
    import platform
    blender_version = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    
    if platform.system() == 'Windows':
        base = Path(os.environ.get('APPDATA', Path.home())) / 'Blender Foundation' / 'Blender' / blender_version
    elif platform.system() == 'Darwin':  # macOS
        base = Path.home() / 'Library' / 'Application Support' / 'Blender' / blender_version
    else:  # Linux
        base = Path.home() / '.config' / 'blender' / blender_version
    
    storage = base / 'bb_waveform_ffmpeg'
    storage.mkdir(parents=True, exist_ok=True)
    return storage


def get_ffmpeg_path():
    """Get the FFmpeg executable path based on priority system"""
    try:
        # Try custom path first
        prefs = bpy.context.preferences.addons[get_addon_name()].preferences
        if prefs.ffmpeg_path and Path(prefs.ffmpeg_path).exists():
            return prefs.ffmpeg_path
    except:
        pass
    
    # Try persistent storage (survives reinstall)
    storage = get_ffmpeg_storage_path()
    if sys.platform.startswith('win'):
        ffmpeg_bin = storage / 'ffmpeg.exe'
    else:
        ffmpeg_bin = storage / 'ffmpeg'
    
    if ffmpeg_bin.exists():
        return str(ffmpeg_bin)
    
    # Try addon folder (legacy location)
    addon_loc = Path(__file__).parent
    if sys.platform.startswith('win'):
        ffmpeg_bin = addon_loc / 'ffmpeg.exe'
    else:
        ffmpeg_bin = addon_loc / 'ffmpeg'
    
    if ffmpeg_bin.exists():
        return str(ffmpeg_bin)
    
    # Default to system PATH
    return 'ffmpeg'


def get_ffprobe_path():
    """Get the FFprobe executable path based on priority system
    Note: If ffprobe isn't available, ffmpeg can handle probing too"""
    try:
        # Try custom path first
        prefs = bpy.context.preferences.addons[get_addon_name()].preferences
        if prefs.ffmpeg_path and Path(prefs.ffmpeg_path).exists():
            probe_dir = Path(prefs.ffmpeg_path).parent
            probe_name = 'ffprobe.exe' if sys.platform.startswith('win') else 'ffprobe'
            ffprobe_path = probe_dir / probe_name
            if ffprobe_path.exists():
                return str(ffprobe_path)
            # Fallback to ffmpeg for probing
            return prefs.ffmpeg_path
    except:
        pass
    
    # Try persistent storage (survives reinstall)
    storage = get_ffmpeg_storage_path()
    if sys.platform.startswith('win'):
        ffprobe_bin = storage / 'ffprobe.exe'
        ffmpeg_bin = storage / 'ffmpeg.exe'
    else:
        ffprobe_bin = storage / 'ffprobe'
        ffmpeg_bin = storage / 'ffmpeg'
    
    if ffprobe_bin.exists():
        return str(ffprobe_bin)
    elif ffmpeg_bin.exists():
        # Use ffmpeg for probing if ffprobe not available
        return str(ffmpeg_bin)
    
    # Try addon folder (legacy location)
    addon_loc = Path(__file__).parent
    if sys.platform.startswith('win'):
        ffprobe_bin = addon_loc / 'ffprobe.exe'
        ffmpeg_bin = addon_loc / 'ffmpeg.exe'
    else:
        ffprobe_bin = addon_loc / 'ffprobe'
        ffmpeg_bin = addon_loc / 'ffmpeg'
    
    if ffprobe_bin.exists():
        return str(ffprobe_bin)
    elif ffmpeg_bin.exists():
        # Use ffmpeg for probing if ffprobe not available
        return str(ffmpeg_bin)
    
    # Try system PATH
    ffprobe_sys = shutil.which('ffprobe')
    if ffprobe_sys:
        return 'ffprobe'
    
    # Fallback to ffmpeg (most builds include probe functionality)
    return 'ffmpeg'


def check_ffmpeg_available():
    """Check if FFmpeg is installed and available (ffprobe is optional)"""
    try:
        ffmpeg_path = get_ffmpeg_path()
        subprocess.run([ffmpeg_path, '-version'], capture_output=True, check=True, timeout=5)
        
        # ffprobe is optional - just check if it exists
        try:
            ffprobe_path = get_ffprobe_path()
            subprocess.run([ffprobe_path, '-version'], capture_output=True, check=True, timeout=5)
        except:
            # ffprobe not available, but that's okay - just won't have multi-track detection
            pass
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def rgb_to_hex(rgb):
    """Convert RGB to hex color string"""
    r, g, b = rgb[:3]
    return ('{:02X}{:02X}{:02X}').format(int(r*255), int(g*255), int(b*255))


def find_empty_channel(seq, start_frame, duration):
    """Find an empty channel in the sequencer that can fit the audio strip"""
    for channel in range(1, 33):
        conflict = False
        for strip in get_sequences(seq):
            if strip.channel == channel:
                if not (strip.frame_final_end <= start_frame or strip.frame_final_start >= start_frame + duration):
                    conflict = True
                    break
        if not conflict:
            return channel
    max_channel = max([s.channel for s in get_sequences(seq)], default=0)
    return max_channel + 1


def get_audio_tracks_info(filepath):
    """Get audio track information including index and codec name"""
    try:
        cmd = [
            get_ffprobe_path(),
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=index,codec_name,channels:stream_tags',
            '-of', 'json',
            str(filepath)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            tracks = []
            for i, stream in enumerate(data.get('streams', [])):
                tags = stream.get('tags', {})
                
                # Try to get a meaningful name from various fields
                name = tags.get('title', '')
                if not name:
                    name = tags.get('handler_name', '')
                if not name:
                    name = tags.get('name', '')
                    
                # Clean up common unhelpful names
                if name in ['SoundHandler', 'Sound Handler', 'Core Media Audio', '']:
                    name = ''
                
                track_info = {
                    'index': i,
                    'codec': stream.get('codec_name', 'unknown'),
                    'channels': stream.get('channels', 0),
                    'language': tags.get('language', ''),
                    'name': name
                }
                tracks.append(track_info)
            
            return tracks
        else:
            # ffprobe failed, try fallback
            return get_audio_tracks_info_fallback(filepath)
    except subprocess.TimeoutExpired:
        print("[WARNING] ffprobe timeout, using fallback")
        return get_audio_tracks_info_fallback(filepath)
    except Exception as e:
        print(f"[WARNING] ffprobe error: {e}, using fallback")
        return get_audio_tracks_info_fallback(filepath)


def get_audio_tracks_info_fallback(filepath):
    """Fallback method using ffmpeg -i to detect audio tracks when ffprobe unavailable"""
    try:
        cmd = [get_ffmpeg_path(), '-i', str(filepath), '-hide_banner']
        # ffmpeg outputs stream info to stderr
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Parse stderr for audio stream info (format: "Stream #0:1: Audio: aac")
        tracks = []
        audio_pattern = r'Stream #\d+:(\d+).*?: Audio: (\w+)'
        
        for match in re.finditer(audio_pattern, result.stderr):
            stream_index = int(match.group(1))
            codec = match.group(2)
            tracks.append({
                'index': stream_index,
                'codec': codec,
                'channels': 2,
                'language': '',
                'name': ''
            })
        
        return tracks
    except Exception as e:
        print(f"[WARNING] Fallback audio track detection failed: {e}")
        return []


def get_audio_track_count(filepath):
    """Get the number of audio tracks in a file"""
    tracks = get_audio_tracks_info(filepath)
    return len(tracks)


def get_audio_duration(filepath):
    """Get exact audio duration in seconds from ffprobe"""
    try:
        cmd = [
            get_ffprobe_path(),
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(filepath)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            duration = data.get('format', {}).get('duration')
            if duration:
                return float(duration)
        return None
    except subprocess.TimeoutExpired:
        print("[WARNING] ffprobe timeout")
        return None
    except Exception as e:
        print(f"[WARNING] ffprobe error: {e}")
        return None


def build_overlay_filter(num_inputs):
    """Build FFmpeg overlay filter chain for compositing multiple waveforms
    
    Returns filter string for overlaying N inputs with screen blend mode.
    Example: [0:v][1:v]overlay[tmp1];[tmp1][2:v]overlay[tmp2];[tmp2][3:v]overlay
    """
    if num_inputs == 1:
        return None  # No filter needed
    
    filter_parts = []
    for i in range(num_inputs - 1):
        if i == 0:
            filter_parts.append('[0:v][1:v]overlay=0:0:format=auto')
        else:
            filter_parts.append(f'[tmp{i}][{i+1}:v]overlay=0:0:format=auto')
        
        # Add output label for all but the last overlay
        if i < num_inputs - 2:
            filter_parts.append(f'[tmp{i+1}];')
    
    return ''.join(filter_parts)


def generate_composite_waveform(waveform_files, output_path):
    """Composite multiple waveform images using FFmpeg overlay
    
    Args:
        waveform_files: List of Path objects to individual waveform PNGs
        output_path: Path where composite should be saved
        
    Returns:
        True if successful, False otherwise
    """
    if len(waveform_files) == 1:
        # Just copy the single file
        import shutil
        shutil.copy(waveform_files[0], output_path)
        return True
    
    # Build input arguments
    inputs = []
    for wf in waveform_files:
        inputs.extend(['-i', str(wf)])
    
    # Build filter chain
    filter_str = build_overlay_filter(len(waveform_files))
    
    cmd = [get_ffmpeg_path()] + inputs + [
        '-hide_banner',
        '-loglevel', 'error',
        '-filter_complex', filter_str,
        '-frames:v', '1',
        '-y', str(output_path)
    ]
    
    try:
        ret = subprocess.call(cmd, timeout=30)
        return ret == 0
    except subprocess.TimeoutExpired:
        print("[ERROR] FFmpeg composite timeout")
        return False
    except Exception as e:
        print(f"[ERROR] FFmpeg composite failed: {e}")
        return False


def generate_multitrack_waveform(filepath, start_frame, width=4000, enabled_tracks=None, track_colors=None, all_red=False, track_order=None):
    """Generate separate waveform images for each enabled audio track with user-selected colors"""
    
    # Get all audio tracks
    all_tracks = get_audio_tracks_info(filepath)
    if not all_tracks:
        print("[MULTITRACK] No tracks found")
        return None, None
    
    # Filter to only enabled tracks
    if enabled_tracks:
        tracks = [t for t in all_tracks if enabled_tracks[t['index']]]
    else:
        tracks = all_tracks
    
    if not tracks:
        print("[MULTITRACK] No tracks enabled")
        return None, None
    
    # Reorder tracks based on track_order if provided
    if track_order:
        track_dict = {t['index']: t for t in tracks}
        reordered_tracks = []
        for track_idx in track_order:
            if track_idx in track_dict:
                reordered_tracks.append(track_dict[track_idx])
        # Add any tracks not in order (shouldn't happen)
        for track in tracks:
            if track not in reordered_tracks:
                reordered_tracks.append(track)
        tracks = reordered_tracks
    
    
    temp_dir = Path(tempfile.gettempdir())
    waveform_files = []
    
    # Generate a waveform for each track
    for i, track in enumerate(tracks):
        # Get color from track_colors if available
        if all_red:
            # Force red when in "All Tracks" mode
            color = (255, 80, 80)
        elif track_colors:
            track_id = str(track['index'])
            color = None
            for item in track_colors:
                if item.name == track_id:
                    # Convert from 0-1 float to 0-255 int
                    color = (int(item.color[0] * 255), int(item.color[1] * 255), int(item.color[2] * 255))
                    break
            if not color:
                color = (255, 80, 80)  # Default red
        else:
            color = (255, 80, 80)  # Default red
        
        hex_color = '{:02X}{:02X}{:02X}'.format(color[0], color[1], color[2])
        
        output_path = temp_dir / f'blender_waveform_track_{i}.png'
        
        # IMPORTANT: Use track['index'] not i for the audio stream index
        track_index = track['index']
        
        cmd = [
            get_ffmpeg_path(),
            '-i', str(filepath),
            '-hide_banner',
            '-loglevel', 'error',
            '-filter_complex',
            f"[0:a:{track_index}]aformat=channel_layouts=mono,showwavespic=s={width}x1000:colors={hex_color}:draw=full,crop=iw:ih/2:0:0",
            '-frames:v', '1',
            '-y', str(output_path)
        ]
        
        try:
            ret = subprocess.call(cmd, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"[WARNING] FFmpeg timeout on track {track_index}")
            continue
        except Exception as e:
            print(f"[WARNING] FFmpeg error on track {track_index}: {e}")
            continue
        
        if ret != 0:
            continue
        
        if output_path.exists():
            waveform_files.append(output_path)
    
    if not waveform_files:
        print("[MULTITRACK] No waveforms generated")
        return None, None
    
    # DON'T reverse - lower in the UI list should be on top
    # User list: [Track1, Track2] means Track1 is top of list, Track2 is bottom of list
    # We want: Track2 on top visually (like Photoshop layers)
    # ffmpeg: [0] is bottom, [N] is top
    # waveform_files is [Track1, Track2], so Track2 will be on top - perfect!
    
    # Composite all waveforms using ffmpeg
    composite_path = temp_dir / 'blender_waveform_mixed.png'
    
    success = generate_composite_waveform(waveform_files, composite_path)
    
    # Clean up individual track files
    for wf in waveform_files:
        try:
            wf.unlink()
        except:
            pass
    
    if not success or not composite_path.exists():
        print("[MULTITRACK] Composite failed")
        return None, None
    
    # Load into Blender
    img_name = 'waveform_temp'
    old_img = bpy.data.images.get(img_name)
    if old_img:
        try:
            old_img.user_clear()
            bpy.data.images.remove(old_img)
        except:
            pass
    
    try:
        img = bpy.data.images.load(str(composite_path), check_existing=False)
        img.name = img_name
    except Exception as e:
        print(f"[ERROR] Failed to load composite image: {e}")
        return None, None
    
    
    return img, composite_path


def generate_multistrip_waveform(strips, start_frame, end_frame, width=4000, strip_colors=None, fps=24.0, all_red=False):
    """Generate waveforms for multiple sequencer strips and composite them"""
    
    temp_dir = Path(tempfile.gettempdir())
    waveform_files = []
    
    # Generate a waveform for each strip
    for i, strip in enumerate(strips):
        # Get color from strip_colors if available
        if all_red:
            # Force red when in "All Tracks" mode
            color = (255, 80, 80)
        elif strip_colors:
            color = None
            for item in strip_colors:
                if item.name == strip.name:
                    # Convert from 0-1 float to 0-255 int
                    color = (int(item.color[0] * 255), int(item.color[1] * 255), int(item.color[2] * 255))
                    break
            if not color:
                color = (255, 80, 80)  # Default red
        else:
            color = (255, 80, 80)  # Default red
        
        hex_color = '{:02X}{:02X}{:02X}'.format(color[0], color[1], color[2])
        
        output_path = temp_dir / f'blender_waveform_strip_{i}.png'
        
        path = bpy.path.abspath(strip.sound.filepath)
        
        cmd = [
            get_ffmpeg_path(),
            '-i', str(path),
            '-hide_banner',
            '-loglevel', 'error',
            '-filter_complex',
            f"[0:a]aformat=channel_layouts=mono,showwavespic=s={width}x1000:colors={hex_color}:draw=full,crop=iw:ih/2:0:0",
            '-frames:v', '1',
            '-y', str(output_path)
        ]
        
        try:
            ret = subprocess.call(cmd, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"[WARNING] FFmpeg timeout on strip {strip.name}")
            continue
        except Exception as e:
            print(f"[WARNING] FFmpeg error on strip {strip.name}: {e}")
            continue
        
        if ret != 0:
            continue
        
        if output_path.exists():
            waveform_files.append(output_path)
    
    if not waveform_files:
        print("[MULTISTRIP] No waveforms generated")
        return None, None
    
    # DON'T reverse - lower in the UI list should be on top
    
    # Composite all waveforms using ffmpeg (same as multitrack)
    composite_path = temp_dir / 'blender_waveform_strips_mixed.png'
    
    success = generate_composite_waveform(waveform_files, composite_path)
    
    # Clean up individual files
    for wf in waveform_files:
        try:
            wf.unlink()
        except:
            pass
    
    if not success or not composite_path.exists():
        print("[MULTISTRIP] Composite failed")
        return None, None
    
    # Load into Blender
    img_name = 'waveform_temp'
    old_img = bpy.data.images.get(img_name)
    if old_img:
        try:
            old_img.user_clear()
            bpy.data.images.remove(old_img)
        except:
            pass
    
    try:
        img = bpy.data.images.load(str(composite_path), check_existing=False)
        img.name = img_name
    except Exception as e:
        print(f"[ERROR] Failed to load composite image: {e}")
        return None, None
    
    
    return img, composite_path


def generate_waveform_image(filepath, start_frame, color=(1, 0.3, 0.3), width=4000, audio_track=0):
    """Generate waveform image using ffmpeg"""
    
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / 'blender_waveform.png'
    
    hex_color = rgb_to_hex(color)
    
    cmd = [
        get_ffmpeg_path(),
        '-i', str(filepath),
        '-hide_banner',
        '-loglevel', 'error',
        '-filter_complex',
        f"[0:a:{audio_track}]aformat=channel_layouts=mono,showwavespic=s={width}x1000:colors={hex_color}:draw=full,crop=iw:ih/2:0:0",
        '-frames:v', '1',
        '-y', str(output_path)
    ]
    
    import time
    start_time = time.time()
    
    try:
        ret = subprocess.call(cmd, timeout=30)
    except subprocess.TimeoutExpired:
        print("[ERROR] FFmpeg waveform generation timeout")
        return None, None
    except Exception as e:
        print(f"[ERROR] FFmpeg waveform generation failed: {e}")
        return None, None
    
    elapsed = time.time() - start_time
    
    if ret != 0:
        print(f"[ERROR] FFmpeg returned error code {ret}")
        return None, None
    
    if not output_path.exists():
        print("[ERROR] Waveform file not created")
        return None, None
    
    img_name = 'waveform_temp'
    
    # Safely remove old image if it exists
    old_img = bpy.data.images.get(img_name)
    if old_img:
        try:
            old_img.user_clear()
            bpy.data.images.remove(old_img)
        except:
            pass
    
    try:
        img = bpy.data.images.load(str(output_path), check_existing=False)
        img.name = img_name
    except Exception as e:
        print(f"[ERROR] Failed to load waveform image: {e}")
        return None, None
    
    
    return img, output_path


def draw_callback(self, context):
    global _waveform_image, _waveform_coords
    
    # Check if image still exists and is valid
    try:
        if not _waveform_image or not _waveform_coords:
            return
        # Try to access the image to see if it's still valid
        _ = _waveform_image.size
    except (ReferenceError, AttributeError) as e:
        # Image has been removed, clear it
        _waveform_image = None
        return
    
    if context.area.type not in ('DOPESHEET_EDITOR', 'GRAPH_EDITOR'):
        return
    
    s = context.scene.waveform_settings
    
    if context.area.type == 'DOPESHEET_EDITOR' and not s.show_ds:
        return
    if context.area.type == 'GRAPH_EDITOR' and not s.show_graph:
        return
    
    if not context.region or not context.region.view2d:
        return
    
    margin = 12 * context.preferences.view.ui_scale
    v2d = context.region.view2d
    
    # Get waveform frame range
    start_frame = _waveform_coords[0][0]
    end_frame = _waveform_coords[1][0]
    
    # Convert start and end frames to screen X coordinates
    x_start = v2d.view_to_region(start_frame, 0, clip=False)[0]
    x_end = v2d.view_to_region(end_frame, 0, clip=False)[0]
    
    width_in_pixels = x_end - x_start
    if width_in_pixels <= 0:
        return
    
    # Calculate height - use a fixed base height scaled by height_offset
    # Don't use image aspect ratio since resolution changes shouldn't affect visual height
    height_in_pixels = BASE_WAVEFORM_HEIGHT * s.height_offset
    
    bottom_y_pixels = margin
    
    # Build coordinates in screen space
    coords = [
        [x_start, bottom_y_pixels],
        [x_end, bottom_y_pixels],
        [x_end, bottom_y_pixels + height_in_pixels],
        [x_start, bottom_y_pixels + height_in_pixels]
    ]
    
    # Draw image
    try:
        shader = gpu.shader.from_builtin('IMAGE')
        batch = batch_for_shader(
            shader, 'TRI_FAN',
            {
                "pos": coords,
                "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
            },
        )
        
        texture = gpu.texture.from_image(_waveform_image)
        gpu.state.blend_set('ADDITIVE')
        shader.uniform_sampler("image", texture)
        shader.bind()
        batch.draw(shader)
        gpu.state.blend_set('NONE')
    except Exception as e:
        print(f"[ERROR] GPU draw failed: {e}")


def clear_handlers():
    global _handlers
    for sp, h in _handlers:
        try:
            sp.draw_handler_remove(h, "WINDOW")
        except:
            pass
    _handlers.clear()


def setup_handlers(context):
    s = context.scene.waveform_settings
    clear_handlers()
    
    
    if not s.enabled:
        return
    
    args = (None, context)
    
    if s.show_ds:
        h = bpy.types.SpaceDopeSheetEditor.draw_handler_add(
            draw_callback, args, "WINDOW", "POST_PIXEL"
        )
        _handlers.append((bpy.types.SpaceDopeSheetEditor, h))
    
    if s.show_graph:
        h = bpy.types.SpaceGraphEditor.draw_handler_add(
            draw_callback, args, "WINDOW", "POST_PIXEL"
        )
        _handlers.append((bpy.types.SpaceGraphEditor, h))
    


def remove_waveform_strips(context):
    """Remove all waveform audio strips from sequencer"""
    if not context.scene.sequence_editor:
        return
    
    seq = context.scene.sequence_editor
    strips_to_remove = []
    for strip in get_sequences(seq):
        if strip.type == 'SOUND' and strip.name.startswith("Waveform Audio"):
            strips_to_remove.append(strip)
    
    # Use the sequences collection directly for removal
    sequences_collection = get_sequences(seq)
    for strip in strips_to_remove:
        # Try different removal methods for compatibility
        if hasattr(sequences_collection, 'remove'):
            sequences_collection.remove(strip)
        elif hasattr(seq, 'sequences') and hasattr(seq.sequences, 'remove'):
            seq.sequences.remove(strip)
        elif hasattr(seq, 'sequences_all') and hasattr(seq.sequences_all, 'remove'):
            seq.sequences_all.remove(strip)


def source_changed(s, context):
    """Called when switching between FILE and SEQ mode"""
    global _waveform_image, _waveform_coords
    
    
    # Clear current waveform display
    _waveform_image = None
    _waveform_coords = None
    clear_handlers()
    
    # If switching to SEQ, remove any FILE mode audio strips
    if s.source == "SEQ":
        remove_waveform_strips(context)
        
        # If no strips are enabled, enable the first one
        if not s.enabled_strips:
            seq = context.scene.sequence_editor
            if seq:
                for strip in get_sequences(seq):
                    if strip.type == 'SOUND':
                        s.enabled_strips = strip.name
                        break
    elif s.source == "FILE":
        # Initialize track colors with random colors if file exists
        if s.filepath:
            path = bpy.path.abspath(s.filepath)
            if Path(path).exists():
                tracks = get_audio_tracks_info(path)
                for track in tracks:
                    track_id = str(track['index'])
                    existing = None
                    for item in s.track_colors:
                        if item.name == track_id:
                            existing = item
                            break
                    if not existing:
                        new_item = s.track_colors.add()
                        new_item.name = track_id
                        new_item.color = get_random_color()
    
    # Rebuild with new mode
    rebuild(context)


def waveform_enabled_changed(s, context):
    """Called when enabling/disabling the waveform"""
    # If enabling in SEQ mode and no strips selected, select the first one
    if s.enabled and s.source == "SEQ" and not s.enabled_strips:
        seq = context.scene.sequence_editor
        if seq:
            for strip in get_sequences(seq):
                if strip.type == 'SOUND':
                    s.enabled_strips = strip.name
                    break
    
    rebuild(context)


def filepath_changed(s, context):
    """Called when the file path changes in FILE mode"""
    if s.source == "FILE" and s.filepath:
        path = bpy.path.abspath(s.filepath)
        if Path(path).exists():
            # Initialize track colors with random colors for any new tracks
            tracks = get_audio_tracks_info(path)
            for track in tracks:
                track_id = str(track['index'])
                existing = None
                for item in s.track_colors:
                    if item.name == track_id:
                        existing = item
                        break
                if not existing:
                    new_item = s.track_colors.add()
                    new_item.name = track_id
                    new_item.color = get_random_color()
    
    rebuild(context)


def resolution_level_update(s, context):
    """Update resolution when level changes"""
    global _rebuilding
    if _rebuilding:
        return
    s.resolution = BASE_WAVEFORM_WIDTH * s.resolution_level
    rebuild(context)


# Using bpy.app.timers instead of threading.Timer (Blender-safe)
_pending_color_update = None

def delayed_color_rebuild():
    """Delayed rebuild function for color updates"""
    global _pending_color_update
    _pending_color_update = None
    # Get context from the window manager
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ('DOPESHEET_EDITOR', 'GRAPH_EDITOR'):
                with bpy.context.temp_override(window=window, area=area):
                    rebuild(bpy.context)
                return None
    return None

def schedule_color_update(context):
    """Schedule a color update using Blender's timer system - will only fire after user stops changing color"""
    global _pending_color_update
    
    # Cancel any existing timer
    if _pending_color_update is not None:
        try:
            bpy.app.timers.unregister(delayed_color_rebuild)
        except:
            pass
    
    # Schedule a new update for COLOR_UPDATE_DELAY seconds from now
    # This way, it only fires when user stops dragging
    _pending_color_update = True
    bpy.app.timers.register(delayed_color_rebuild, first_interval=COLOR_UPDATE_DELAY)


def get_random_color():
    """Generate colors with evenly distributed hues using golden ratio"""
    global _color_index
    
    # Use golden ratio to distribute hues evenly
    golden_ratio = 0.618033988749895
    hue = (_color_index * golden_ratio) % 1.0
    _color_index += 1
    
    # High saturation and value for vibrant colors
    saturation = 0.8
    value = 1.0
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b)


class BB_TrackColorItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        default=(1.0, 0.3, 0.3),
        min=0.0,
        max=1.0,
        size=3,
        update=lambda s, c: schedule_color_update(c)
    )


class BB_StripColorItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        default=(1.0, 0.3, 0.3),
        min=0.0,
        max=1.0,
        size=3,
        update=lambda s, c: schedule_color_update(c)
    )


def rebuild(context):
    global _waveform_image, _waveform_coords, _rebuilding
    
    # Prevent recursive calls
    if _rebuilding:
        return
    
    _rebuilding = True
    
    try:
        s = context.scene.waveform_settings
        
        # Sync resolution_level with resolution (in case resolution was changed directly)
        if s.resolution >= BASE_WAVEFORM_WIDTH:
            s.resolution_level = max(1, s.resolution // BASE_WAVEFORM_WIDTH)
        
        
        if not s.enabled:
            clear_handlers()
            _waveform_image = None
            _waveform_coords = None
            remove_waveform_strips(context)
            return
        
        # Get audio file path and duration
        path = None
        start_frame = None
        sw_frames = None
        color = (1.0, 0.3, 0.3)  # Default color
        
        if s.source == "SEQ":
            seq = context.scene.sequence_editor
            if not seq:
                return
            
            # Parse enabled strips
            enabled_names = set(s.enabled_strips.split(',')) if s.enabled_strips else set()
            enabled_names.discard('')
            
            
            # Get all sound strips - create a dict for lookup
            all_sound_strips = []
            strip_dict = {}
            for strip in get_sequences(seq):
                if strip.type == 'SOUND':
                    all_sound_strips.append(strip)
                    strip_dict[strip.name] = strip
            
            # Get enabled strips in the correct order using strip_order
            enabled_sound_strips = []
            if s.strip_order:
                order_names = s.strip_order.split(',')
                for name in order_names:
                    if name in enabled_names and name in strip_dict:
                        enabled_sound_strips.append(strip_dict[name])
            else:
                # Fallback if no strip_order
                for strip in all_sound_strips:
                    if strip.name in enabled_names:
                        enabled_sound_strips.append(strip)
            
            
            # If no valid strips are enabled but we have sound strips, enable the first one
            if not enabled_sound_strips and all_sound_strips:
                first_strip = all_sound_strips[0]
                s.enabled_strips = first_strip.name
                enabled_sound_strips = [first_strip]
            
            if not enabled_sound_strips:
                return
            
            # If multiple strips, use multitrack mode
            if len(enabled_sound_strips) > 1:
                
                # Reorder strips based on strip_order
                if s.strip_order:
                    order_names = s.strip_order.split(',')
                    strip_dict = {strip.name: strip for strip in enabled_sound_strips}
                    reordered_strips = []
                    for name in order_names:
                        if name in strip_dict:
                            reordered_strips.append(strip_dict[name])
                    # Add any strips not in order_names (shouldn't happen, but just in case)
                    for strip in enabled_sound_strips:
                        if strip not in reordered_strips:
                            reordered_strips.append(strip)
                    enabled_sound_strips = reordered_strips
                
                # Find the earliest start and latest end
                start_frame = min(s.frame_final_start for s in enabled_sound_strips)
                end_frame = max(s.frame_final_end for s in enabled_sound_strips)
                sw_frames = end_frame - start_frame
                
                # Calculate width based on frame duration
                pixels_per_frame = s.resolution_level * PIXELS_PER_FRAME_PER_LEVEL
                waveform_width = int(sw_frames * pixels_per_frame)
                
                # Generate multistrip composite
                fps = context.scene.render.fps / context.scene.render.fps_base
                img, img_path = generate_multistrip_waveform(enabled_sound_strips, start_frame, end_frame, 
                                                            width=waveform_width, strip_colors=s.strip_colors,
                                                            fps=fps, all_red=s.all_channels_red)
                if not img:
                    print("ERROR generating multistrip waveform")
                    return
                
                _waveform_image = img
                
                _waveform_coords = (
                    (start_frame, 0),
                    (start_frame + sw_frames, 0),
                    (start_frame + sw_frames, BASE_WAVEFORM_HEIGHT),
                    (start_frame, BASE_WAVEFORM_HEIGHT)
                )
                
                
                setup_handlers(context)
                
                for area in context.screen.areas:
                    area.tag_redraw()
                
                return
            else:
                # Single strip mode
                strip = enabled_sound_strips[0]
                path = bpy.path.abspath(strip.sound.filepath)
                start_frame = strip.frame_final_start
                end_frame = strip.frame_final_end
                sw_frames = end_frame - start_frame
                
                # Get the color for this strip
                if s.all_channels_red:
                    color = (1.0, 0.3, 0.3)  # Force red in all tracks mode
                else:
                    color = (1.0, 0.3, 0.3)  # Default red
                    for item in s.strip_colors:
                        if item.name == strip.name:
                            color = tuple(item.color)
                            break
                
        else:
            path = bpy.path.abspath(s.filepath)
            start_frame = s.start_frame
            
            
            # Remove old waveform strips first
            remove_waveform_strips(context)
            
            # Try to find if user has this file in sequencer already (for duration calculation only)
            user_strip_with_same_file = None
            if context.scene.sequence_editor:
                seq = context.scene.sequence_editor
                for strip in get_sequences(seq):
                    if strip.type == 'SOUND':
                        strip_path = bpy.path.abspath(strip.sound.filepath)
                        if strip_path == path:
                            user_strip_with_same_file = strip
                            break
            
            if user_strip_with_same_file:
                # Use the user's strip duration
                sw_frames = user_strip_with_same_file.frame_final_end - user_strip_with_same_file.frame_final_start
            else:
                # Get duration from ffprobe for accuracy
                fps = context.scene.render.fps / context.scene.render.fps_base
                duration_seconds = get_audio_duration(path)
                
                if duration_seconds:
                    sw_frames = round(duration_seconds * fps)
                else:
                    # Fallback to Blender's calculation
                    try:
                        sound = bpy.data.sounds.load(path, check_existing=True)
                        
                        # Try to get duration - API varies by Blender version
                        if hasattr(sound, 'duration'):
                            duration_seconds = sound.duration
                        elif hasattr(sound, 'length'):
                            duration_seconds = sound.length
                        else:
                            # Fallback: create a temp strip to get duration
                            if not context.scene.sequence_editor:
                                context.scene.sequence_editor_create()
                            temp_seq = context.scene.sequence_editor
                            
                            # Use proper API to create strip
                            sequences = get_sequences(temp_seq)
                            if hasattr(sequences, 'new_sound'):
                                temp_strip = sequences.new_sound("_temp_", path, 1, 1)
                            else:
                                # Fallback for Blender 5.0
                                temp_strip = temp_seq.sequences.new_sound("_temp_", path, 1, 1)
                            
                            duration_seconds = (temp_strip.frame_final_end - temp_strip.frame_final_start) / fps
                            
                            # Remove temp strip using proper API
                            if hasattr(sequences, 'remove'):
                                sequences.remove(temp_strip)
                            elif hasattr(temp_seq, 'sequences') and hasattr(temp_seq.sequences, 'remove'):
                                temp_seq.sequences.remove(temp_strip)
                        
                        sw_frames = round(duration_seconds * fps)
                    except Exception as e:
                        print(f"Error loading sound: {e}")
                        import traceback
                        traceback.print_exc()
                        sw_frames = 250  # fallback
            
            # Always add our own strip to sequencer for audio playback
            if not context.scene.sequence_editor:
                context.scene.sequence_editor_create()
            
            seq = context.scene.sequence_editor
            empty_channel = find_empty_channel(seq, start_frame, sw_frames)
            
            try:
                sequences = get_sequences(seq)
                if hasattr(sequences, 'new_sound'):
                    strip = sequences.new_sound("Waveform Audio", path, empty_channel, start_frame)
                else:
                    # Fallback for Blender 5.0
                    strip = seq.sequences.new_sound("Waveform Audio", path, empty_channel, start_frame)
                
                # Update sw_frames with the actual strip duration
                sw_frames = strip.frame_final_end - strip.frame_final_start
            except Exception as e:
                print(f"Error adding to sequencer: {e}")
        
        
        if not Path(path).exists():
            print("ERROR — File does NOT exist")
            return
        
        
        # Check if we should use multitrack mode (only for FILE mode with multiple audio tracks)
        if s.source == "FILE":
            available_tracks = get_audio_track_count(path)
            enabled_count = sum(1 for i in range(available_tracks) if s.enabled_tracks[i])
            
            if enabled_count == 0:
                print("[WARNING] No tracks enabled, using track 0")
                s.enabled_tracks[0] = True
                enabled_count = 1
            
            if enabled_count > 1:
                # Calculate width based on frame duration and resolution level
                # This ensures the waveform pixel width matches the timeline frame range exactly
                pixels_per_frame = s.resolution_level * PIXELS_PER_FRAME_PER_LEVEL
                waveform_width = int(sw_frames * pixels_per_frame)
                
                # Generate mixed multi-track waveform
                img, img_path = generate_multitrack_waveform(path, start_frame, width=waveform_width, 
                                                             enabled_tracks=s.enabled_tracks,
                                                             track_colors=s.track_colors,
                                                             all_red=s.all_tracks_red,
                                                             track_order=list(s.track_order))
            else:
                # Single track mode - find which track is enabled and get its color
                audio_track_to_use = 0
                for i in range(available_tracks):
                    if s.enabled_tracks[i]:
                        audio_track_to_use = i
                        break
                
                # Calculate width based on frame duration
                pixels_per_frame = s.resolution_level * PIXELS_PER_FRAME_PER_LEVEL
                waveform_width = int(sw_frames * pixels_per_frame)
                
                # Get the color for this track
                if s.all_tracks_red:
                    color = (1.0, 0.3, 0.3)  # Force red in all tracks mode
                else:
                    color = (1.0, 0.3, 0.3)  # Default red
                    track_id = str(audio_track_to_use)
                    for item in s.track_colors:
                        if item.name == track_id:
                            color = tuple(item.color)
                            break
                
                # Generate single track waveform
                img, img_path = generate_waveform_image(path, start_frame, color=color, width=waveform_width, audio_track=audio_track_to_use)
        else:
            # SEQ mode - single strip waveform (color already set)
            # Calculate width based on frame duration
            pixels_per_frame = s.resolution_level * PIXELS_PER_FRAME_PER_LEVEL
            waveform_width = int(sw_frames * pixels_per_frame)
            
            # Generate waveform
            img, img_path = generate_waveform_image(path, start_frame, color=color, width=waveform_width, audio_track=0)
        
        if not img:
            return
        
        
        _waveform_image = img
        
        _waveform_coords = (
            (start_frame, 0),
            (start_frame + sw_frames, 0),
            (start_frame + sw_frames, BASE_WAVEFORM_HEIGHT),
            (start_frame, BASE_WAVEFORM_HEIGHT)
        )
        
        
        setup_handlers(context)
        
        
        for area in context.screen.areas:
            area.tag_redraw()
        
    finally:
        _rebuilding = False


class BB_WaveformSettings(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Enable Waveform",
        default=False,
        update=waveform_enabled_changed
    )
    
    source: bpy.props.EnumProperty(
        name="Audio Source",
        items=[
            ('SEQ', 'Sequencer', 'Use audio from sequencer strips'),
            ('FILE', 'File', 'Use external audio file')
        ],
        default='FILE',
        update=source_changed
    )
    
    filepath: bpy.props.StringProperty(
        name="Audio File",
        subtype='FILE_PATH',
        update=filepath_changed
    )
    
    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        default=1,
        min=0,
        update=lambda s, c: rebuild(c)
    )
    
    show_ds: bpy.props.BoolProperty(
        name="Show in Dope Sheet",
        default=True,
        update=lambda s, c: setup_handlers(c)
    )
    
    show_graph: bpy.props.BoolProperty(
        name="Show in Graph Editor",
        default=True,
        update=lambda s, c: setup_handlers(c)
    )
    
    height_offset: bpy.props.FloatProperty(
        name="Height",
        default=1.0,
        min=0.1,
        max=5.0,
        update=lambda s, c: [area.tag_redraw() for area in c.screen.areas]
    )
    
    resolution: bpy.props.IntProperty(
        name="Resolution",
        default=BASE_WAVEFORM_WIDTH,
        min=1000,
        max=32000,
        update=lambda s, c: rebuild(c)
    )
    
    resolution_level: bpy.props.IntProperty(
        name="Resolution Level",
        description="Multiplier for base resolution (4000 pixels)",
        default=1,
        min=1,
        max=8,
        update=resolution_level_update
    )
    
    # Multi-strip support (for SEQ mode)
    enabled_strips: bpy.props.StringProperty(
        name="Enabled Strips",
        default="",
        update=lambda s, c: rebuild(c)
    )
    
    strip_order: bpy.props.StringProperty(
        name="Strip Order",
        description="Order of strips for display (comma-separated strip names)",
        default=""
    )
    
    strip_colors: bpy.props.CollectionProperty(type=BB_StripColorItem)
    
    all_channels_red: bpy.props.BoolProperty(
        name="All Tracks Red",
        description="Force all sequencer strips to display in red",
        default=True,
        update=lambda s, c: rebuild(c)
    )
    
    # Multi-track support (for FILE mode)
    enabled_tracks: bpy.props.BoolVectorProperty(
        name="Enabled Tracks",
        size=MAX_AUDIO_TRACKS,
        default=[True] + [False] * (MAX_AUDIO_TRACKS - 1)
    )
    
    track_order: bpy.props.IntVectorProperty(
        name="Track Order",
        description="Order of tracks for display",
        size=MAX_AUDIO_TRACKS,
        default=tuple(range(MAX_AUDIO_TRACKS))
    )
    
    track_colors: bpy.props.CollectionProperty(type=BB_TrackColorItem)
    
    all_tracks_red: bpy.props.BoolProperty(
        name="All Tracks Red",
        description="Force all audio tracks to display in red",
        default=True,
        update=lambda s, c: rebuild(c)
    )


class BB_PT_dopesheet(bpy.types.Panel):
    bl_label = "BB Waveform"
    bl_idname = "BB_PT_dopesheet"
    bl_space_type = 'DOPESHEET_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'BB Waveform'
    
    # No poll method - show in all DOPESHEET_EDITOR modes that have sidebars
    
    def draw(self, context):
        layout = self.layout
        s = context.scene.waveform_settings
        
        # Check for FFmpeg on first draw
        if not check_ffmpeg_available():
            box = layout.box()
            box.alert = True
            col = box.column(align=True)
            col.label(text="FFmpeg Not Found!", icon='ERROR')
            col.label(text="Install FFmpeg to use this addon")
            
            row = col.row(align=True)
            row.operator('bb_waveform.download_ffmpeg', text='Auto-install', icon='IMPORT')
            row.operator('bb_waveform.check_ffmpeg', text='Check Status', icon='INFO')
            
            return
        
        layout.prop(s, "enabled", toggle=True)
        
        if not s.enabled:
            return
        
        layout.separator()
        
        # Source selection
        layout.prop(s, "source", expand=True)
        
        layout.separator()
        
        if s.source == "FILE":
            # FILE mode settings
            layout.prop(s, "filepath")
            
            if s.filepath:
                path = bpy.path.abspath(s.filepath)
                if Path(path).exists():
                    # Check if file has multiple tracks
                    available_tracks = get_audio_track_count(path)
                    
                    if available_tracks > 1:
                        # Multi-track file
                        box = layout.box()
                        box.label(text=f"Multi-track audio ({available_tracks} tracks)", icon='SOUND')
                        
                        row = box.row()
                        row.prop(s, "all_tracks_red", text="All Tracks", toggle=True)
                        row.operator("waveform.select_audio_track", text="Select Tracks", icon='RESTRICT_SELECT_OFF')
                    else:
                        # Single track file
                        layout.label(text="Single track audio", icon='SOUND')
                else:
                    layout.label(text="File not found", icon='ERROR')
            
            layout.separator()
            
            row = layout.row()
            row.prop(s, "start_frame")
            
        else:
            # SEQ mode settings
            seq = context.scene.sequence_editor
            if not seq:
                layout.label(text="No sequencer", icon='ERROR')
                return
            
            sound_strips = [s for s in get_sequences(seq) if s.type == 'SOUND']
            
            if not sound_strips:
                layout.label(text="No audio strips in sequencer", icon='INFO')
            else:
                if len(sound_strips) > 1:
                    box = layout.box()
                    box.label(text=f"Multiple audio strips ({len(sound_strips)})", icon='SOUND')
                    
                    row = box.row()
                    row.prop(s, "all_channels_red", text="All Tracks", toggle=True)
                    row.operator("waveform.select_channel", text="Select Tracks", icon='RESTRICT_SELECT_OFF')
                else:
                    layout.label(text="Single audio strip", icon='SOUND')
        
        layout.separator()
        
        # Display settings
        box = layout.box()
        box.label(text="Display", icon='RESTRICT_VIEW_OFF')
        box.prop(s, "show_ds", text="Dope Sheet / Timeline")
        box.prop(s, "show_graph", text="Graph Editor")
        box.prop(s, "height_offset")
        
        layout.separator()
        
        # Resolution settings
        box = layout.box()
        box.label(text="Resolution", icon='IMAGE_DATA')
        box.prop(s, "resolution_level", text="Level")


class BB_PT_graph(bpy.types.Panel):
    bl_label = "BB Waveform"
    bl_idname = "BB_PT_graph"
    bl_space_type = 'GRAPH_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'BB Waveform'
    
    def draw(self, context):
        layout = self.layout
        s = context.scene.waveform_settings
        
        # Check for FFmpeg on first draw
        if not check_ffmpeg_available():
            box = layout.box()
            box.alert = True
            col = box.column(align=True)
            col.label(text="FFmpeg Not Found!", icon='ERROR')
            col.label(text="Install FFmpeg to use this addon")
            
            row = col.row(align=True)
            row.operator('bb_waveform.download_ffmpeg', text='Auto-install', icon='IMPORT')
            row.operator('bb_waveform.check_ffmpeg', text='Check Status', icon='INFO')
            
            return
        
        layout.prop(s, "enabled", toggle=True)
        
        if not s.enabled:
            return
        
        layout.separator()
        
        # Source selection
        layout.prop(s, "source", expand=True)
        
        layout.separator()
        
        if s.source == "FILE":
            # FILE mode settings
            layout.prop(s, "filepath")
            
            if s.filepath:
                path = bpy.path.abspath(s.filepath)
                if Path(path).exists():
                    # Check if file has multiple tracks
                    available_tracks = get_audio_track_count(path)
                    
                    if available_tracks > 1:
                        # Multi-track file
                        box = layout.box()
                        box.label(text=f"Multi-track audio ({available_tracks} tracks)", icon='SOUND')
                        
                        row = box.row()
                        row.prop(s, "all_tracks_red", text="All Tracks", toggle=True)
                        row.operator("waveform.select_audio_track", text="Select Tracks", icon='RESTRICT_SELECT_OFF')
                    else:
                        # Single track file
                        layout.label(text="Single track audio", icon='SOUND')
                else:
                    layout.label(text="File not found", icon='ERROR')
            
            layout.separator()
            
            row = layout.row()
            row.prop(s, "start_frame")
            
        else:
            # SEQ mode settings
            seq = context.scene.sequence_editor
            if not seq:
                layout.label(text="No sequencer", icon='ERROR')
                return
            
            sound_strips = [s for s in get_sequences(seq) if s.type == 'SOUND']
            
            if not sound_strips:
                layout.label(text="No audio strips in sequencer", icon='INFO')
            else:
                if len(sound_strips) > 1:
                    box = layout.box()
                    box.label(text=f"Multiple audio strips ({len(sound_strips)})", icon='SOUND')
                    
                    row = box.row()
                    row.prop(s, "all_channels_red", text="All Tracks", toggle=True)
                    row.operator("waveform.select_channel", text="Select Tracks", icon='RESTRICT_SELECT_OFF')
                else:
                    layout.label(text="Single audio strip", icon='SOUND')
        
        layout.separator()
        
        # Display settings
        box = layout.box()
        box.label(text="Display", icon='RESTRICT_VIEW_OFF')
        box.prop(s, "show_ds", text="Dope Sheet / Timeline")
        box.prop(s, "show_graph", text="Graph Editor")
        box.prop(s, "height_offset")
        
        layout.separator()
        
        # Resolution settings
        box = layout.box()
        box.label(text="Resolution", icon='IMAGE_DATA')
        box.prop(s, "resolution_level", text="Level")



class BB_OT_select_channel(bpy.types.Operator):
    """Select which audio strips to display and customize their colors"""
    bl_idname = "waveform.select_channel"
    bl_label = "Select Tracks"
    
    def get_strips(self, context):
        seq = context.scene.sequence_editor
        if not seq:
            return []
        
        strips = []
        for strip in get_sequences(seq):
            if strip.type == 'SOUND':
                strips.append(strip)
        
        return strips
    
    def invoke(self, context, event):
        strips = self.get_strips(context)
        
        if not strips:
            self.report({'ERROR'}, 'No audio strips found')
            return {'CANCELLED'}
        
        if len(strips) == 1:
            self.report({'INFO'}, 'Only one audio strip available')
            return {'CANCELLED'}
        
        # Initialize colors for strips that don't have them yet
        s = context.scene.waveform_settings
        
        # Only create color entries for new strips - preserve all existing colors
        # This MUST happen BEFORE clearing the all_channels_red flag
        for strip in strips:
            existing = None
            for item in s.strip_colors:
                if item.name == strip.name:
                    existing = item
                    break
            if not existing:
                new_item = s.strip_colors.add()
                new_item.name = strip.name
                new_item.color = get_random_color()
        
        # Clear the all_channels_red flag - we're in select mode now
        s.all_channels_red = False
        
        # Rebuild waveform immediately with the new colors
        rebuild(context)
        
        return context.window_manager.invoke_popup(self, width=400)
    
    def draw(self, context):
        layout = self.layout
        strips = self.get_strips(context)
        s = context.scene.waveform_settings
        
        layout.label(text="Select strips to display:", icon='SOUND')
        layout.separator()
        
        # Parse enabled strips
        enabled_names = set(s.enabled_strips.split(',')) if s.enabled_strips else set()
        enabled_names.discard('')
        
        # Reorder strips based on strip_order
        strip_order = s.strip_order.split(',') if s.strip_order else []
        ordered_strips = []
        strip_dict = {strip.name: strip for strip in strips}
        
        for name in strip_order:
            if name in strip_dict:
                ordered_strips.append(strip_dict[name])
        
        # Add strips not in order
        for strip in strips:
            if strip not in ordered_strips:
                ordered_strips.append(strip)
        
        # Create rows with checkbox and color picker
        col = layout.column(align=True)
        for i, strip in enumerate(ordered_strips):
            row = col.row(align=True)
            
            # Compact move buttons - DOWN on left, UP on right
            if i < len(ordered_strips) - 1:
                op = row.operator("waveform.move_strip", text="", icon='TRIA_DOWN', emboss=False)
                op.strip_position = i
                op.direction = 'DOWN'
            else:
                row.label(text="", icon='BLANK1')
            
            if i > 0:
                op = row.operator("waveform.move_strip", text="", icon='TRIA_UP', emboss=False)
                op.strip_position = i
                op.direction = 'UP'
            else:
                row.label(text="", icon='BLANK1')
            
            # Checkbox button
            is_enabled = strip.name in enabled_names
            op = row.operator("waveform.toggle_strip", text=strip.name, 
                            depress=is_enabled, emboss=True)
            op.strip_name = strip.name
            
            # Color picker
            for item in s.strip_colors:
                if item.name == strip.name:
                    row.prop(item, "color", text="")
                    break
    
    def execute(self, context):
        return {'FINISHED'}


class BB_OT_toggle_strip(bpy.types.Operator):
    bl_idname = "waveform.toggle_strip"
    bl_label = "Toggle Strip"
    bl_options = {'INTERNAL'}
    
    strip_name: bpy.props.StringProperty()
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        # Parse current enabled strips
        enabled_names = set(s.enabled_strips.split(',')) if s.enabled_strips else set()
        enabled_names.discard('')
        
        # Don't allow disabling the last strip
        if len(enabled_names) == 1 and self.strip_name in enabled_names:
            self.report({'WARNING'}, 'At least one strip must be enabled')
            return {'CANCELLED'}
        
        # Toggle this strip
        if self.strip_name in enabled_names:
            enabled_names.remove(self.strip_name)
        else:
            enabled_names.add(self.strip_name)
        
        # Update property
        s.enabled_strips = ','.join(enabled_names)
        
        # Rebuild immediately
        rebuild(context)
        return {'FINISHED'}


class BB_OT_move_strip(bpy.types.Operator):
    bl_idname = "waveform.move_strip"
    bl_label = "Move Strip"
    bl_options = {'INTERNAL'}
    
    strip_position: bpy.props.IntProperty()  # Position in the display list
    direction: bpy.props.StringProperty()  # 'UP' or 'DOWN'
    
    def execute(self, context):
        s = context.scene.waveform_settings
        seq = context.scene.sequence_editor
        
        if not seq:
            return {'CANCELLED'}
        
        # Get all sound strips
        strips = [strip for strip in get_sequences(seq) if strip.type == 'SOUND']
        
        if not strips:
            return {'CANCELLED'}
        
        # Get strip order - if empty, initialize with current strip order
        strip_order = s.strip_order.split(',') if s.strip_order else []
        strip_order = [n for n in strip_order if n]  # Remove empty strings
        
        # Initialize strip_order if empty
        if not strip_order:
            strip_order = [strip.name for strip in strips]
        
        # Build ordered list
        strip_dict = {strip.name: strip for strip in strips}
        ordered_strips = []
        for name in strip_order:
            if name in strip_dict:
                ordered_strips.append(strip_dict[name])
        
        # Add any missing strips
        for strip in strips:
            if strip not in ordered_strips:
                ordered_strips.append(strip)
        
        if self.strip_position < 0 or self.strip_position >= len(ordered_strips):
            return {'CANCELLED'}
        
        # Swap positions
        if self.direction == 'UP' and self.strip_position > 0:
            ordered_strips[self.strip_position], ordered_strips[self.strip_position - 1] = \
                ordered_strips[self.strip_position - 1], ordered_strips[self.strip_position]
        elif self.direction == 'DOWN' and self.strip_position < len(ordered_strips) - 1:
            ordered_strips[self.strip_position], ordered_strips[self.strip_position + 1] = \
                ordered_strips[self.strip_position + 1], ordered_strips[self.strip_position]
        else:
            return {'CANCELLED'}
        
        # Update strip_order
        s.strip_order = ','.join(strip.name for strip in ordered_strips)
        
        # Rebuild to update waveform with new order
        rebuild(context)
        return {'FINISHED'}


class BB_OT_select_audio_track(bpy.types.Operator):
    """Select which audio tracks to display and customize their colors"""
    bl_idname = "waveform.select_audio_track"
    bl_label = "Select Tracks"
    
    def get_audio_tracks(self, context):
        s = context.scene.waveform_settings
        
        if s.source == "FILE" and s.filepath:
            path = bpy.path.abspath(s.filepath)
        else:
            seq = context.scene.sequence_editor
            if seq and seq.active_strip and seq.active_strip.type == "SOUND":
                path = bpy.path.abspath(seq.active_strip.sound.filepath)
            else:
                return []
        
        if not Path(path).exists():
            return []
        
        return get_audio_tracks_info(path)
    
    def invoke(self, context, event):
        tracks = self.get_audio_tracks(context)
        
        if not tracks:
            self.report({'ERROR'}, 'No audio tracks found in file')
            return {'CANCELLED'}
        
        if len(tracks) == 1:
            self.report({'INFO'}, 'File has only one audio track')
            return {'CANCELLED'}
        
        # Initialize colors for tracks that don't have them yet
        s = context.scene.waveform_settings
        
        # Only create color entries for new tracks - preserve all existing colors
        # This MUST happen BEFORE clearing the all_tracks_red flag
        for track in tracks:
            track_id = str(track['index'])
            existing = None
            for item in s.track_colors:
                if item.name == track_id:
                    existing = item
                    break
            if not existing:
                new_item = s.track_colors.add()
                new_item.name = track_id
                new_item.color = get_random_color()
        
        # Clear the all_tracks_red flag - we're in select mode now
        s.all_tracks_red = False
        
        # Rebuild waveform immediately with the new colors
        rebuild(context)
        
        return context.window_manager.invoke_popup(self, width=400)
    
    def draw(self, context):
        layout = self.layout
        tracks = self.get_audio_tracks(context)
        s = context.scene.waveform_settings
        
        layout.label(text="Select tracks to display:", icon='SOUND')
        layout.separator()
        
        # Reorder tracks based on track_order
        track_order = list(s.track_order)
        ordered_tracks = []
        for track_idx in track_order[:len(tracks)]:
            for track in tracks:
                if track['index'] == track_idx:
                    ordered_tracks.append(track)
                    break
        
        # Create rows with checkbox and color picker
        col = layout.column(align=True)
        for i, track in enumerate(ordered_tracks):
            row = col.row(align=True)
            
            # Compact move buttons - DOWN on left, UP on right
            if i < len(ordered_tracks) - 1:
                op = row.operator("waveform.move_track", text="", icon='TRIA_DOWN', emboss=False)
                op.track_position = i
                op.direction = 'DOWN'
            else:
                row.label(text="", icon='BLANK1')
            
            if i > 0:
                op = row.operator("waveform.move_track", text="", icon='TRIA_UP', emboss=False)
                op.track_position = i
                op.direction = 'UP'
            else:
                row.label(text="", icon='BLANK1')
            
            # Checkbox button
            is_enabled = s.enabled_tracks[track['index']]
            op = row.operator("waveform.toggle_track", text=f"Track {track['index'] + 1}", 
                            depress=is_enabled, emboss=True)
            op.track_index = track['index']
            
            # Color picker
            track_id = str(track['index'])
            for item in s.track_colors:
                if item.name == track_id:
                    row.prop(item, "color", text="")
                    break
    
    def execute(self, context):
        return {'FINISHED'}


class BB_OT_toggle_track(bpy.types.Operator):
    bl_idname = "waveform.toggle_track"
    bl_label = "Toggle Track"
    bl_options = {'INTERNAL'}
    
    track_index: bpy.props.IntProperty()
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        # Get actual track count from the file
        if s.source == "FILE":
            path = bpy.path.abspath(s.filepath)
        else:
            seq = context.scene.sequence_editor
            if seq and seq.active_strip and seq.active_strip.type == "SOUND":
                path = bpy.path.abspath(seq.active_strip.sound.filepath)
            else:
                # Can't determine track count, allow toggle
                s.enabled_tracks[self.track_index] = not s.enabled_tracks[self.track_index]
                rebuild(context)
                return {'FINISHED'}
        
        if not Path(path).exists():
            # Can't check file, allow toggle
            s.enabled_tracks[self.track_index] = not s.enabled_tracks[self.track_index]
            rebuild(context)
            return {'FINISHED'}
        
        # Count only the tracks that exist in the file
        available_tracks = get_audio_track_count(path)
        enabled_count = sum(1 for i in range(available_tracks) if s.enabled_tracks[i])
        
        if enabled_count == 1 and s.enabled_tracks[self.track_index]:
            # Don't allow disabling the last track
            self.report({'WARNING'}, 'At least one track must be enabled')
            return {'CANCELLED'}
        
        # Toggle the track
        s.enabled_tracks[self.track_index] = not s.enabled_tracks[self.track_index]
        
        
        # Rebuild immediately
        rebuild(context)
        return {'FINISHED'}


class BB_OT_move_track(bpy.types.Operator):
    bl_idname = "waveform.move_track"
    bl_label = "Move Track"
    bl_options = {'INTERNAL'}
    
    track_position: bpy.props.IntProperty()  # Position in the display list
    direction: bpy.props.StringProperty()  # 'UP' or 'DOWN'
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        # Get track order as a list
        track_order = list(s.track_order)
        
        # Get available tracks count
        if s.source == "FILE" and s.filepath:
            path = bpy.path.abspath(s.filepath)
            if Path(path).exists():
                tracks = get_audio_tracks_info(path)
                num_tracks = len(tracks)
            else:
                return {'CANCELLED'}
        else:
            return {'CANCELLED'}
        
        # Only work with the portion of track_order that corresponds to actual tracks
        active_order = track_order[:num_tracks]
        
        if self.track_position < 0 or self.track_position >= len(active_order):
            return {'CANCELLED'}
        
        # Swap positions based on direction
        if self.direction == 'UP' and self.track_position > 0:
            active_order[self.track_position], active_order[self.track_position - 1] = \
                active_order[self.track_position - 1], active_order[self.track_position]
        elif self.direction == 'DOWN' and self.track_position < len(active_order) - 1:
            active_order[self.track_position], active_order[self.track_position + 1] = \
                active_order[self.track_position + 1], active_order[self.track_position]
        else:
            return {'CANCELLED'}
        
        # Update track_order with the new arrangement
        track_order[:num_tracks] = active_order
        s.track_order = track_order
        
        # Rebuild to update waveform with new order
        rebuild(context)
        return {'FINISHED'}


# FFmpeg Management Operators and Preferences
# Insert this code BEFORE the classes tuple

class BB_OT_download_ffmpeg(bpy.types.Operator):
    """Download and install FFmpeg in the addon folder"""
    bl_idname = "bb_waveform.download_ffmpeg"
    bl_label = "Auto-install FFmpeg"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def invoke(self, context, event):
        # Determine platform-specific download URL
        if sys.platform.startswith('win'):
            # Note: These URLs from Pullusb repository may not have ffprobe
            # If ffprobe fails, the addon will still work with ffmpeg only
            self.release_url = 'https://github.com/Pullusb/static_bin/raw/main/ffmpeg/windows/ffmpeg.exe'
            self.probe_url = 'https://github.com/Pullusb/static_bin/raw/main/ffmpeg/windows/ffprobe.exe'
        elif sys.platform.startswith(('linux', 'freebsd')):
            self.release_url = 'https://github.com/Pullusb/static_bin/raw/main/ffmpeg/linux/ffmpeg'
            self.probe_url = 'https://github.com/Pullusb/static_bin/raw/main/ffmpeg/linux/ffprobe'
        else:
            self.release_url = 'https://github.com/Pullusb/static_bin/raw/main/ffmpeg/mac/ffmpeg'
            self.probe_url = 'https://github.com/Pullusb/static_bin/raw/main/ffmpeg/mac/ffprobe'
        
        # Use persistent storage that survives addon reinstall
        storage_loc = get_ffmpeg_storage_path()
        self.ffmpeg_bin = storage_loc / Path(self.release_url).name
        self.ffprobe_bin = storage_loc / Path(self.probe_url).name
        self.exists = self.ffmpeg_bin.exists() and self.ffprobe_bin.exists()
        
        return context.window_manager.invoke_props_dialog(self, width=500)
    
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        if self.exists:
            col.label(text='FFmpeg is already in addon folder, delete and re-download? (80-100 MB)', icon='INFO')
        else:
            col.label(text='This will download FFmpeg static release in addon folder (80-100 MB)', icon='INFO')
            col.label(text='Would you like to continue?')
    
    def execute(self, context):
        if self.exists:
            try:
                self.ffmpeg_bin.unlink()
                self.ffprobe_bin.unlink()
            except:
                pass
        
        if sys.platform.startswith(('linux', 'freebsd')):
            ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            # Download ffmpeg
            self.report({'INFO'}, 'Downloading ffmpeg...')
            with urllib.request.urlopen(self.release_url, timeout=60) as response, open(self.ffmpeg_bin, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
            if not self.ffmpeg_bin.exists():
                self.report({'ERROR'}, 'ffmpeg download failed')
                return {'CANCELLED'}
            
            # Try to download ffprobe (optional - ffmpeg includes probing functionality)
            self.report({'INFO'}, 'Downloading ffprobe...')
            try:
                with urllib.request.urlopen(self.probe_url, timeout=60) as response, open(self.ffprobe_bin, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as probe_err:
                # This is fine - ffmpeg can handle probing on its own
                pass
            
            # Set permissions on Unix
            if not sys.platform.startswith('win'):
                import os
                if self.ffmpeg_bin.exists():
                    os.chmod(self.ffmpeg_bin, 0o755)
                if self.ffprobe_bin.exists():
                    os.chmod(self.ffprobe_bin, 0o755)
            
            # Report success
            if self.ffmpeg_bin.exists():
                prefs = context.preferences.addons[get_addon_name()].preferences
                prefs.ffmpeg_path = str(self.ffmpeg_bin.resolve())
                self.report({'INFO'}, 'FFmpeg installed successfully!')
            else:
                self.report({'ERROR'}, 'Download failed')
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f'Download failed: {str(e)}')
            return {'CANCELLED'}
        
        return {'FINISHED'}


class BB_OT_check_ffmpeg(bpy.types.Operator):
    """Check FFmpeg availability"""
    bl_idname = "bb_waveform.check_ffmpeg"
    bl_label = "Check FFmpeg Status"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def invoke(self, context, event):
        prefs = context.preferences.addons[get_addon_name()].preferences
        
        self.sys_ffmpeg = shutil.which('ffmpeg')
        self.sys_ffprobe = shutil.which('ffprobe')
        
        self.custom_ffmpeg = None
        self.custom_ffprobe = None
        if prefs.ffmpeg_path and Path(prefs.ffmpeg_path).exists():
            self.custom_ffmpeg = prefs.ffmpeg_path
            custom_dir = Path(prefs.ffmpeg_path).parent
            probe_name = 'ffprobe.exe' if sys.platform.startswith('win') else 'ffprobe'
            probe_path = custom_dir / probe_name
            if probe_path.exists():
                self.custom_ffprobe = str(probe_path)
        
        # Check persistent storage location
        storage_loc = get_ffmpeg_storage_path()
        if sys.platform.startswith('win'):
            storage_ffmpeg = storage_loc / 'ffmpeg.exe'
            storage_ffprobe = storage_loc / 'ffprobe.exe'
        else:
            storage_ffmpeg = storage_loc / 'ffmpeg'
            storage_ffprobe = storage_loc / 'ffprobe'
        
        self.storage_ffmpeg = str(storage_ffmpeg) if storage_ffmpeg.exists() else None
        self.storage_ffprobe = str(storage_ffprobe) if storage_ffprobe.exists() else None
        
        return context.window_manager.invoke_props_dialog(self, width=550)
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text='FFmpeg Search Priority:', icon='INFO')
        layout.label(text='1. Custom Path → 2. Persistent Storage → 3. System PATH')
        layout.separator()
        
        if self.custom_ffmpeg:
            box = layout.box()
            col = box.column()
            col.label(text='✓ Custom Path (WILL BE USED)', icon='CHECKMARK')
            col.label(text=f'  ffmpeg: {self.custom_ffmpeg}')
            if self.custom_ffprobe:
                col.label(text=f'  ffprobe: {self.custom_ffprobe}')
            else:
                col.label(text='  ffprobe: NOT FOUND', icon='ERROR')
        
        if self.storage_ffmpeg:
            box = layout.box()
            col = box.column()
            if not self.custom_ffmpeg:
                col.label(text='✓ Persistent Storage (WILL BE USED)', icon='CHECKMARK')
            else:
                col.label(text='✓ Persistent Storage (available)', icon='DISCLOSURE_TRI_RIGHT')
            col.label(text=f'  ffmpeg: {self.storage_ffmpeg}')
            if self.storage_ffprobe:
                col.label(text=f'  ffprobe: {self.storage_ffprobe}')
            else:
                col.label(text='  ffprobe: NOT FOUND', icon='ERROR')
        
        if self.sys_ffmpeg:
            box = layout.box()
            col = box.column()
            if not self.custom_ffmpeg and not self.storage_ffmpeg:
                col.label(text='✓ System PATH (WILL BE USED)', icon='CHECKMARK')
            else:
                col.label(text='✓ System PATH (available)', icon='DISCLOSURE_TRI_RIGHT')
            col.label(text=f'  ffmpeg: {self.sys_ffmpeg}')
            if self.sys_ffprobe:
                col.label(text=f'  ffprobe: {self.sys_ffprobe}')
            else:
                col.label(text='  ffprobe: NOT FOUND', icon='ERROR')
        
        if not self.custom_ffmpeg and not self.storage_ffmpeg and not self.sys_ffmpeg:
            box = layout.box()
            box.alert = True
            col = box.column()
            col.label(text='✗ FFmpeg NOT FOUND', icon='ERROR')
            col.label(text='Use Auto-install or manually install FFmpeg')
    
    def execute(self, context):
        return {'FINISHED'}


class BB_OT_remove_ffmpeg(bpy.types.Operator):
    """Remove FFmpeg from persistent storage"""
    bl_idname = "bb_waveform.remove_ffmpeg"
    bl_label = "Remove FFmpeg"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def invoke(self, context, event):
        storage_loc = get_ffmpeg_storage_path()
        if sys.platform.startswith('win'):
            self.ffmpeg_bin = storage_loc / 'ffmpeg.exe'
            self.ffprobe_bin = storage_loc / 'ffprobe.exe'
        else:
            self.ffmpeg_bin = storage_loc / 'ffmpeg'
            self.ffprobe_bin = storage_loc / 'ffprobe'
        
        self.exists = self.ffmpeg_bin.exists() or self.ffprobe_bin.exists()
        
        return context.window_manager.invoke_props_dialog(self, width=500)
    
    def draw(self, context):
        layout = self.layout
        if self.exists:
            layout.label(text='Remove FFmpeg from persistent storage?', icon='ERROR')
            if self.ffmpeg_bin.exists():
                layout.label(text=f'  {self.ffmpeg_bin}')
            if self.ffprobe_bin.exists():
                layout.label(text=f'  {self.ffprobe_bin}')
        else:
            layout.label(text='No FFmpeg found in storage', icon='INFO')
    
    def execute(self, context):
        removed = []
        try:
            if self.ffmpeg_bin.exists():
                self.ffmpeg_bin.unlink()
                removed.append('ffmpeg')
            if self.ffprobe_bin.exists():
                self.ffprobe_bin.unlink()
                removed.append('ffprobe')
            
            if removed:
                self.report({'INFO'}, f"Removed: {', '.join(removed)}")
            else:
                self.report({'INFO'}, 'Nothing to remove')
                
        except Exception as e:
            self.report({'ERROR'}, f'Could not remove files: {str(e)}')
            return {'CANCELLED'}
        
        return {'FINISHED'}


class BB_OT_open_preferences(bpy.types.Operator):
    """Open addon preferences"""
    bl_idname = "bb_waveform.open_preferences"
    bl_label = "Open Preferences"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def execute(self, context):
        bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        context.preferences.active_section = 'ADDONS'
        
        # For extensions, this doesn't work the same way, but at least opens preferences
        return {'FINISHED'}


class BB_WaveformPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    
    ffmpeg_path: bpy.props.StringProperty(
        name="Custom FFmpeg Path",
        description="Path to ffmpeg executable (leave empty to use addon folder or system PATH)",
        subtype='FILE_PATH',
        default=""
    )
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        
        box = layout.box()
        box.label(text='FFmpeg Installation:', icon='PLUGIN')
        
        col = box.column()
        col.label(text='This addon requires FFmpeg to generate waveforms')
        
        row = col.row()
        row.operator('bb_waveform.check_ffmpeg', text='Check Status', icon='INFO')
        
        row = col.row(align=True)
        row.operator('bb_waveform.download_ffmpeg', text='Auto-install', icon='IMPORT')
        row.operator('bb_waveform.remove_ffmpeg', text='Remove', icon='X')
        
        col.label(text='Or specify custom FFmpeg path:')
        col.prop(self, 'ffmpeg_path', text='')
        
        col.label(text='Priority: Custom Path > Persistent Storage > System PATH', icon='INFO')


classes = (BB_TrackColorItem, BB_StripColorItem, BB_WaveformSettings, 
           BB_PT_dopesheet, BB_PT_graph, 
           BB_OT_select_channel, BB_OT_toggle_strip, BB_OT_move_strip,
           BB_OT_select_audio_track, BB_OT_toggle_track, BB_OT_move_track,
           BB_OT_download_ffmpeg, BB_OT_check_ffmpeg, BB_OT_remove_ffmpeg,
           BB_OT_open_preferences,
           BB_WaveformPreferences)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.waveform_settings = bpy.props.PointerProperty(type=BB_WaveformSettings)


def unregister():
    global _pending_color_update
    
    # Cancel any pending color update timers
    if _pending_color_update is not None:
        try:
            bpy.app.timers.unregister(delayed_color_rebuild)
        except:
            pass
        _pending_color_update = None
    
    clear_handlers()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.waveform_settings
