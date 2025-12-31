import bpy, gpu, subprocess, tempfile
from gpu_extras.batch import batch_for_shader
from pathlib import Path
import colorsys

bl_info = {
    "name": "BB Waveform",
    "author": "Blender Bob & Claude.ai",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > BB Waveform",
    "description": "Display audio waveforms in timeline, dopesheet, and graph editors",
    "category": "Animation",
}

_handlers = []
_waveform_image = None
_waveform_coords = None
_rebuilding = False  # Flag to prevent recursive rebuilds


def rgb_to_hex(rgb):
    """Convert RGB to hex color string"""
    r, g, b = rgb[:3]
    return ('{:02X}{:02X}{:02X}').format(int(r*255), int(g*255), int(b*255))


def find_empty_channel(seq, start_frame, duration):
    """Find an empty channel in the sequencer that can fit the audio strip"""
    for channel in range(1, 33):
        conflict = False
        for strip in seq.sequences:
            if strip.channel == channel:
                if not (strip.frame_final_end <= start_frame or strip.frame_final_start >= start_frame + duration):
                    conflict = True
                    break
        if not conflict:
            return channel
    max_channel = max([s.channel for s in seq.sequences], default=0)
    return max_channel + 1


def get_audio_tracks_info(filepath):
    """Get audio track information including index and codec name"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=index,codec_name,channels:stream_tags',
            '-of', 'json',
            str(filepath)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
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
            
            print(f"[AUDIO TRACKS] Found {len(tracks)} audio track(s)")
            return tracks
        else:
            print(f"[AUDIO TRACKS] Could not detect tracks")
            return []
    except Exception as e:
        print(f"[AUDIO TRACKS] Error detecting: {e}")
        return []


def get_audio_track_count(filepath):
    """Get the number of audio tracks in a file"""
    tracks = get_audio_tracks_info(filepath)
    return len(tracks)


def generate_multitrack_waveform(filepath, start_frame, width=4000, enabled_tracks=None, track_colors=None, all_red=False):
    """Generate separate waveform images for each enabled audio track with user-selected colors"""
    print(f"[MULTITRACK] Creating mixed waveform for: {filepath}")
    
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
    
    print(f"[MULTITRACK] Processing {len(tracks)} enabled track(s)")
    
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
        
        cmd = [
            'ffmpeg',
            '-i', str(filepath),
            '-hide_banner',
            '-loglevel', 'error',
            '-filter_complex',
            f"[0:a:{i}]aformat=channel_layouts=mono,showwavespic=s={width}x1000:colors={hex_color}:draw=full,crop=iw:ih/2:0:0",
            '-frames:v', '1',
            '-y', str(output_path)
        ]
        
        print(f"[MULTITRACK] Generating track {i+1} with color #{hex_color}...")
        ret = subprocess.call(cmd)
        
        if ret != 0:
            print(f"[MULTITRACK] Warning: Failed to generate track {i}")
            continue
        
        if output_path.exists():
            waveform_files.append(output_path)
    
    if not waveform_files:
        print("[MULTITRACK] No waveforms generated")
        return None, None
    
    # Composite all waveforms using ffmpeg
    composite_path = temp_dir / 'blender_waveform_mixed.png'
    
    # Build ffmpeg filter for overlaying all tracks with screen blend mode
    if len(waveform_files) == 1:
        # Just copy the single file
        waveform_files[0].replace(composite_path)
    else:
        # Create overlay filter chain
        inputs = []
        for wf in waveform_files:
            inputs.extend(['-i', str(wf)])
        
        # Build filter: overlay each subsequent image with screen blend
        filter_parts = ['[0:v]']
        for i in range(1, len(waveform_files)):
            filter_parts.append(f'[{i}:v]overlay=0:0:format=auto')
            if i < len(waveform_files) - 1:
                filter_parts.append(',')
        
        filter_str = ''.join(filter_parts)
        
        cmd = ['ffmpeg'] + inputs + [
            '-hide_banner',
            '-loglevel', 'error',
            '-filter_complex', filter_str,
            '-frames:v', '1',
            '-y', str(composite_path)
        ]
        
        print(f"[MULTITRACK] Compositing {len(waveform_files)} tracks...")
        ret = subprocess.call(cmd)
        
        if ret != 0:
            print("[MULTITRACK] Composite failed")
            return None, None
    
    # Clean up individual track files
    for wf in waveform_files:
        try:
            wf.unlink()
        except:
            pass
    
    if not composite_path.exists():
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
    
    img = bpy.data.images.load(str(composite_path), check_existing=False)
    img.name = img_name
    
    print(f"[MULTITRACK] Success! Mixed waveform size: {img.size[0]}x{img.size[1]}")
    
    return img, composite_path


def generate_multistrip_waveform(strips, start_frame, end_frame, width=4000, strip_colors=None, fps=24.0, all_red=False):
    """Generate waveforms for multiple sequencer strips and composite them"""
    print(f"[MULTISTRIP] Creating mixed waveform for {len(strips)} strips")
    
    temp_dir = Path(tempfile.gettempdir())
    waveform_files = []
    
    # Generate a waveform for each strip
    for i, strip in enumerate(strips):
        # Get color from strip_colors if available
        if all_red:
            # Force red when in "All Channels" mode
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
            'ffmpeg',
            '-i', str(path),
            '-hide_banner',
            '-loglevel', 'error',
            '-filter_complex',
            f"[0:a]aformat=channel_layouts=mono,showwavespic=s={width}x1000:colors={hex_color}:draw=full,crop=iw:ih/2:0:0",
            '-frames:v', '1',
            '-y', str(output_path)
        ]
        
        print(f"[MULTISTRIP] Generating strip '{strip.name}' with color #{hex_color}...")
        ret = subprocess.call(cmd)
        
        if ret != 0:
            print(f"[MULTISTRIP] Warning: Failed to generate strip {strip.name}")
            continue
        
        if output_path.exists():
            waveform_files.append(output_path)
    
    if not waveform_files:
        print("[MULTISTRIP] No waveforms generated")
        return None, None
    
    # Composite all waveforms using ffmpeg (same as multitrack)
    composite_path = temp_dir / 'blender_waveform_strips_mixed.png'
    
    if len(waveform_files) == 1:
        # Just copy the single file
        import shutil
        shutil.copy(waveform_files[0], composite_path)
    else:
        # Create overlay filter chain (same as multitrack)
        inputs = []
        for wf in waveform_files:
            inputs.extend(['-i', str(wf)])
        
        # Build filter: overlay each subsequent image
        filter_parts = ['[0:v]']
        for i in range(1, len(waveform_files)):
            filter_parts.append(f'[{i}:v]overlay=0:0:format=auto')
            if i < len(waveform_files) - 1:
                filter_parts.append(',')
        
        filter_str = ''.join(filter_parts)
        
        cmd = ['ffmpeg'] + inputs + [
            '-hide_banner',
            '-loglevel', 'error',
            '-filter_complex', filter_str,
            '-frames:v', '1',
            '-y', str(composite_path)
        ]
        
        print(f"[MULTISTRIP] Compositing {len(waveform_files)} strips...")
        ret = subprocess.call(cmd)
        
        if ret != 0:
            print("[MULTISTRIP] Composite failed")
            return None, None
    
    # Clean up individual files
    for wf in waveform_files:
        try:
            wf.unlink()
        except:
            pass
    
    if not composite_path.exists():
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
    
    img = bpy.data.images.load(str(composite_path), check_existing=False)
    img.name = img_name
    
    print(f"[MULTISTRIP] Success! Mixed waveform size: {img.size[0]}x{img.size[1]}")
    
    return img, composite_path


def generate_waveform_image(filepath, start_frame, color=(1, 0.3, 0.3), width=4000, audio_track=0):
    """Generate waveform image using ffmpeg"""
    print(f"[GENERATE] Creating waveform for: {filepath}")
    print(f"[GENERATE] Resolution: {width}x1000")
    print(f"[GENERATE] Audio track: {audio_track}")
    print(f"[GENERATE] This may take a moment for large files...")
    
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / 'blender_waveform.png'
    
    hex_color = rgb_to_hex(color)
    
    cmd = [
        'ffmpeg',
        '-i', str(filepath),
        '-hide_banner',
        '-loglevel', 'error',
        '-filter_complex',
        f"[0:a:{audio_track}]aformat=channel_layouts=mono,showwavespic=s={width}x1000:colors={hex_color}:draw=full,crop=iw:ih/2:0:0",
        '-frames:v', '1',
        '-y', str(output_path)
    ]
    
    print(f"[FFMPEG] Processing audio...")
    import time
    start_time = time.time()
    
    ret = subprocess.call(cmd)
    
    elapsed = time.time() - start_time
    print(f"[FFMPEG] Completed in {elapsed:.2f} seconds")
    
    if ret != 0:
        print(f"[ERROR] ffmpeg failed with code {ret}")
        return None, None
    
    if not output_path.exists():
        print(f"[ERROR] Output not created at {output_path}")
        return None, None
    
    img_name = 'waveform_temp'
    
    # Safely remove old image if it exists
    old_img = bpy.data.images.get(img_name)
    if old_img:
        try:
            old_img.user_clear()
            bpy.data.images.remove(old_img)
            print(f"[GENERATE] Removed old waveform image")
        except:
            pass
    
    img = bpy.data.images.load(str(output_path), check_existing=False)
    img.name = img_name
    
    print(f"[GENERATE] Success! Image size: {img.size[0]}x{img.size[1]}")
    
    return img, output_path


def draw_callback(self, context):
    global _waveform_image, _waveform_coords
    
    # Check if image still exists and is valid
    try:
        if not _waveform_image or not _waveform_coords:
            return
        # Try to access the image to see if it's still valid
        _ = _waveform_image.size
    except (ReferenceError, AttributeError):
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
    
    # Calculate height based on image aspect ratio and scale
    if _waveform_image.size[0] > 0:
        aspect_ratio = _waveform_image.size[1] / _waveform_image.size[0]
        height_in_pixels = width_in_pixels * aspect_ratio * s.height_offset
    else:
        height_in_pixels = 100 * s.height_offset
    
    bottom_y_pixels = margin
    
    # Build coordinates in screen space
    coords = [
        [x_start, bottom_y_pixels],
        [x_end, bottom_y_pixels],
        [x_end, bottom_y_pixels + height_in_pixels],
        [x_start, bottom_y_pixels + height_in_pixels]
    ]
    
    # Draw image
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


def clear_handlers():
    global _handlers
    for sp, h in _handlers:
        try:
            sp.draw_handler_remove(h, "WINDOW")
        except:
            pass
    _handlers.clear()
    print("[HANDLER] cleared")


def setup_handlers(context):
    print("[HANDLER] setting up")
    s = context.scene.waveform_settings
    clear_handlers()
    
    if not s.enabled:
        print("[HANDLER] disabled")
        return
    
    args = (None, context)
    
    if s.show_ds:
        print("  adding Dope/Timeline")
        h = bpy.types.SpaceDopeSheetEditor.draw_handler_add(
            draw_callback, args, "WINDOW", "POST_PIXEL"
        )
        _handlers.append((bpy.types.SpaceDopeSheetEditor, h))
    
    if s.show_graph:
        print("  adding Graph Editor")
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
    for strip in seq.sequences:
        if strip.type == 'SOUND' and strip.name.startswith("Waveform Audio"):
            strips_to_remove.append(strip)
    
    for strip in strips_to_remove:
        seq.sequences.remove(strip)
        print(f"Removed waveform audio strip")


def source_changed(s, context):
    """Called when switching between FILE and SEQ mode"""
    global _waveform_image, _waveform_coords
    
    print(f"[SOURCE CHANGE] Switching to {s.source} mode")
    
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
                for strip in seq.sequences:
                    if strip.type == 'SOUND':
                        s.enabled_strips = strip.name
                        print(f"[SOURCE CHANGE] Auto-enabled first strip: {strip.name}")
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
                        print(f"[SOURCE CHANGE] Initialized track {track_id} with random color")
    
    # Rebuild with new mode
    rebuild(context)


def waveform_enabled_changed(s, context):
    """Called when enabling/disabling the waveform"""
    # If enabling in SEQ mode and no strips selected, select the first one
    if s.enabled and s.source == "SEQ" and not s.enabled_strips:
        seq = context.scene.sequence_editor
        if seq:
            for strip in seq.sequences:
                if strip.type == 'SOUND':
                    s.enabled_strips = strip.name
                    print(f"[ENABLE] Auto-enabled first strip: {strip.name}")
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
                    print(f"[FILEPATH CHANGE] Initialized track {track_id} with random color")
    
    rebuild(context)


def resolution_level_update(s, context):
    """Update resolution when level changes"""
    global _rebuilding
    if _rebuilding:
        return
    s.resolution = 4000 * s.resolution_level
    rebuild(context)


def rebuild(context):
    global _waveform_image, _waveform_coords, _rebuilding


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


_color_update_timer = None


def schedule_color_update(context):
    """Schedule a color update - will only fire after user stops changing color"""
    global _color_update_timer
    
    # Cancel any existing timer
    if _color_update_timer is not None and _color_update_timer.is_alive:
        _color_update_timer.cancel()
    
    # Schedule a new update for 0.3 seconds from now
    # This way, it only fires when user stops dragging
    import threading
    _color_update_timer = threading.Timer(0.3, lambda: rebuild(context))
    _color_update_timer.start()


_color_index = 0  # Track which color to assign next


def get_random_color():
    """Generate colors with evenly distributed hues"""
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


def rebuild(context):
    global _waveform_image, _waveform_coords, _rebuilding
    
    # Prevent recursive calls
    if _rebuilding:
        return
    
    _rebuilding = True
    
    try:
        print("\n=== REBUILD ===")
        s = context.scene.waveform_settings
        
        # Sync resolution_level with resolution (in case resolution was changed directly)
        if s.resolution >= 4000:
            s.resolution_level = max(1, s.resolution // 4000)
        
        if not s.enabled:
            print("Disabled — clearing")
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
                print("No sequence editor")
                return
            
            # Parse enabled strips
            enabled_names = set(s.enabled_strips.split(',')) if s.enabled_strips else set()
            enabled_names.discard('')
            
            print(f"[SEQ MODE DEBUG] enabled_strips string: '{s.enabled_strips}'")
            print(f"[SEQ MODE DEBUG] enabled_names set: {enabled_names}")
            
            # Get all sound strips that are enabled
            enabled_sound_strips = []
            all_sound_strips = []
            for strip in seq.sequences:
                if strip.type == 'SOUND':
                    all_sound_strips.append(strip)
                    if strip.name in enabled_names:
                        enabled_sound_strips.append(strip)
            
            print(f"[SEQ MODE DEBUG] Found {len(enabled_sound_strips)} enabled strips out of {len(all_sound_strips)} total sound strips")
            
            # If no valid strips are enabled but we have sound strips, enable the first one
            if not enabled_sound_strips and all_sound_strips:
                first_strip = all_sound_strips[0]
                s.enabled_strips = first_strip.name
                enabled_sound_strips = [first_strip]
                print(f"[SEQ MODE] No valid strips enabled, auto-enabled: {first_strip.name}")
            
            if not enabled_sound_strips:
                print("No enabled sound strips")
                return
            
            # If multiple strips, use multitrack mode
            if len(enabled_sound_strips) > 1:
                print(f"[SEQ MODE] Multiple strips mode: {len(enabled_sound_strips)} strips")
                # Find the earliest start and latest end
                start_frame = min(s.frame_start for s in enabled_sound_strips)
                end_frame = max(s.frame_final_end for s in enabled_sound_strips)
                sw_frames = end_frame - start_frame
                
                # Generate multistrip composite
                img, img_path = generate_multistrip_waveform(enabled_sound_strips, start_frame, end_frame, 
                                                            width=s.resolution, strip_colors=s.strip_colors,
                                                            all_red=s.all_channels_red)
                if not img:
                    print("ERROR generating multistrip waveform")
                    return
                
                _waveform_image = img
                
                _waveform_coords = (
                    (start_frame, 0),
                    (start_frame + sw_frames, 0),
                    (start_frame + sw_frames, 100),
                    (start_frame, 100)
                )
                
                print(f"[COORDS] Start={start_frame}, End={start_frame + sw_frames}, Duration={sw_frames} frames")
                
                setup_handlers(context)
                
                for area in context.screen.areas:
                    area.tag_redraw()
                
                return
            else:
                # Single strip mode
                strip = enabled_sound_strips[0]
                path = bpy.path.abspath(strip.sound.filepath)
                start_frame = strip.frame_start
                end_frame = strip.frame_final_end
                sw_frames = end_frame - start_frame
                
                # Get the color for this strip
                if s.all_channels_red:
                    color = (1.0, 0.3, 0.3)  # Force red in all channels mode
                else:
                    color = (1.0, 0.3, 0.3)  # Default red
                    for item in s.strip_colors:
                        if item.name == strip.name:
                            color = tuple(item.color)
                            break
                
                print(f"[SEQ MODE] Using sequencer strip: {strip.name}")
                print(f"[SEQ MODE] Start: {start_frame}, End: {end_frame}, Duration: {sw_frames} frames")
                print(f"[SEQ MODE] Path: {path}")
        else:
            path = bpy.path.abspath(s.filepath)
            start_frame = s.start_frame
            
            print(f"Using external file @ frame {start_frame}")
            
            # Remove old strips first
            remove_waveform_strips(context)
            
            # Try to find the actual strip duration if it exists in sequencer
            existing_strip = None
            if context.scene.sequence_editor:
                seq = context.scene.sequence_editor
                for strip in seq.sequences:
                    if strip.type == 'SOUND':
                        strip_path = bpy.path.abspath(strip.sound.filepath)
                        if strip_path == path:
                            existing_strip = strip
                            break
            
            if existing_strip:
                # Use the actual strip's duration
                sw_frames = existing_strip.frame_final_end - existing_strip.frame_final_start
                print(f"Found existing strip in sequencer, using its duration: {sw_frames} frames")
            else:
                # Calculate from audio file
                try:
                    sound = bpy.data.sounds.load(path, check_existing=True)
                    # Get correct FPS accounting for fps_base
                    fps = context.scene.render.fps / context.scene.render.fps_base
                    
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
                        temp_strip = temp_seq.sequences.new_sound("_temp_", path, 1, 1)
                        duration_seconds = (temp_strip.frame_final_end - temp_strip.frame_final_start) / fps
                        temp_seq.sequences.remove(temp_strip)
                    
                    sw_frames = round(duration_seconds * fps)
                    print(f"[DEBUG] Sound duration: {duration_seconds:.4f}s")
                    print(f"[DEBUG] Scene FPS: {context.scene.render.fps} / {context.scene.render.fps_base} = {fps}")
                    print(f"[DEBUG] Calculated frames: {sw_frames}")
                except Exception as e:
                    print(f"Error loading sound: {e}")
                    import traceback
                    traceback.print_exc()
                    sw_frames = 250  # fallback
            
            # Add to sequencer for audio playback
            if not existing_strip:
                if not context.scene.sequence_editor:
                    context.scene.sequence_editor_create()
                
                seq = context.scene.sequence_editor
                empty_channel = find_empty_channel(seq, start_frame, sw_frames)
                
                try:
                    strip = seq.sequences.new_sound("Waveform Audio", path, empty_channel, start_frame)
                    # Update sw_frames with the actual strip duration
                    sw_frames = strip.frame_final_end - strip.frame_final_start
                    print(f"Added sound to sequencer on channel {empty_channel}, actual duration: {sw_frames} frames")
                except Exception as e:
                    print(f"Error adding to sequencer: {e}")
        
        print(f"Resolved path: {path}")
        
        if not Path(path).exists():
            print("ERROR — File does NOT exist")
            return
        
        print(f"\n{'='*60}")
        print(f"GENERATING WAVEFORM - Please wait...")
        print(f"Resolution: {s.resolution}px")
        print(f"{'='*60}\n")
        
        # Check if we should use multitrack mode (only for FILE mode with multiple audio tracks)
        if s.source == "FILE":
            available_tracks = get_audio_track_count(path)
            enabled_count = sum(1 for i in range(available_tracks) if s.enabled_tracks[i])
            
            if enabled_count == 0:
                print("[WARNING] No tracks enabled, using track 0")
                s.enabled_tracks[0] = True
                enabled_count = 1
            
            if enabled_count > 1:
                # Generate mixed multi-track waveform
                img, img_path = generate_multitrack_waveform(path, start_frame, width=s.resolution, 
                                                             enabled_tracks=s.enabled_tracks,
                                                             track_colors=s.track_colors,
                                                             all_red=s.all_tracks_red)
            else:
                # Single track mode - find which track is enabled and get its color
                audio_track_to_use = 0
                for i in range(available_tracks):
                    if s.enabled_tracks[i]:
                        audio_track_to_use = i
                        break
                
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
                img, img_path = generate_waveform_image(path, start_frame, color=color, width=s.resolution, audio_track=audio_track_to_use)
        else:
            # SEQ mode - single strip waveform (color already set)
            img, img_path = generate_waveform_image(path, start_frame, color=color, width=s.resolution, audio_track=0)
        
        if not img:
            print("ERROR generating waveform")
            return
        
        _waveform_image = img
        
        # Set up coordinates for drawing with actual frame duration
        _waveform_coords = (
            (start_frame, 0),
            (start_frame + sw_frames, 0),
            (start_frame + sw_frames, 100),
            (start_frame, 100)
        )
        
        print(f"[COORDS] Start={start_frame}, End={start_frame + sw_frames}, Duration={sw_frames} frames")
        
        setup_handlers(context)
        
        # Redraw all areas
        for area in context.screen.areas:
            area.tag_redraw()
    
    finally:
        _rebuilding = False


class BB_WaveformSettings(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Enable Waveform",
        default=False,
        update=lambda s, c: waveform_enabled_changed(s, c)
    )
    source: bpy.props.EnumProperty(
        name="Source",
        items=[("SEQ", "Sequencer", ""), ("FILE", "File", "")],
        default="SEQ",
        update=lambda s, c: source_changed(s, c)
    )
    filepath: bpy.props.StringProperty(
        name="Audio File",
        subtype="FILE_PATH",
        update=lambda s, c: filepath_changed(s, c)
    )
    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        default=1,
        update=lambda s, c: rebuild(c)
    )
    height_offset: bpy.props.FloatProperty(
        name="Height Scale",
        default=1.0,
        min=0.1,
        max=10.0,
        soft_min=0.5,
        soft_max=10.0
    )
    show_ds: bpy.props.BoolProperty(
        name="Dopesheet/Timeline",
        default=True,
        update=lambda s, c: setup_handlers(c)
    )
    show_graph: bpy.props.BoolProperty(
        name="Graph Editor",
        default=False,
        update=lambda s, c: setup_handlers(c)
    )
    add_to_sequencer: bpy.props.BoolProperty(
        name="Add to Sequencer (for audio playback)",
        description="Add the audio file to the sequencer so it plays during animation",
        default=True,
        update=lambda s, c: rebuild(c)
    )
    resolution: bpy.props.IntProperty(
        name="Resolution",
        description="Waveform image width in pixels (internal)",
        default=4000,
        min=1000,
        max=32000
    )
    resolution_level: bpy.props.IntProperty(
        name="Resolution Level",
        description="Resolution quality level (1=4000px, 2=8000px, etc.)",
        default=1,
        min=1,
        max=8,
        update=lambda s, c: resolution_level_update(s, c)
    )
    strip_channel: bpy.props.IntProperty(
        name="Channel",
        description="Sequencer channel to use for waveform",
        default=1,
        min=1,
        max=32
    )
    audio_track: bpy.props.IntProperty(
        name="Audio Track",
        description="Which audio track to use from the file (0=first, 1=second, etc.)",
        default=0,
        min=-1,  # -1 = all tracks mixed
        max=15,
        update=lambda s, c: rebuild(c)
    )
    enabled_tracks: bpy.props.BoolVectorProperty(
        name="Enabled Tracks",
        description="Which tracks to show in the waveform",
        size=16,
        default=[True] + [False] * 15,
        update=lambda s, c: rebuild(c)  # Add update callback
    )
    enabled_strips: bpy.props.StringProperty(
        name="Enabled Strips",
        description="Comma-separated list of enabled strip names",
        default=""
    )
    all_channels_red: bpy.props.BoolProperty(
        name="All Channels Red Mode",
        default=False,
        description="When true, all sequencer strips display in red"
    )
    all_tracks_red: bpy.props.BoolProperty(
        name="All Tracks Red Mode",
        default=False,
        description="When true, all audio tracks display in red"
    )
    track_colors: bpy.props.CollectionProperty(
        type=BB_TrackColorItem,
        name="Track Colors"
    )
    strip_colors: bpy.props.CollectionProperty(
        type=BB_StripColorItem,
        name="Strip Colors"
    )


def draw_ui(self, ctx):
    s = ctx.scene.waveform_settings
    layout = self.layout
    
    layout.prop(s, "enabled")
    
    layout.separator()
    layout.label(text="Show in:")
    layout.prop(s, "show_ds")
    layout.prop(s, "show_graph")
    
    if s.enabled:
        layout.separator()
        layout.prop(s, "source", expand=True)
        
        if s.source == "SEQ":
            row = layout.row(align=True)
            row.operator("waveform.all_channels", text="All Tracks", icon='ALIGN_JUSTIFY')
            row.operator("waveform.select_channel", text="Select Tracks", icon='PRESET')
        
        if s.source == "FILE":
            layout.prop(s, "filepath")
            layout.prop(s, "start_frame")
            row = layout.row(align=True)
            row.operator("waveform.all_tracks", text="All Tracks", icon='ALIGN_JUSTIFY')
            row.operator("waveform.select_audio_track", text="Select Tracks", icon='PRESET')
        
        layout.separator()
        layout.prop(s, "height_offset", slider=True, text="Height Scale")
        
        # Show resolution level with built-in +/- spinners
        layout.prop(s, "resolution_level")


class BB_PT_dopesheet(bpy.types.Panel):
    bl_label = "BB Waveform"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "BB Waveform"
    draw = draw_ui


class BB_PT_graph(bpy.types.Panel):
    bl_label = "BB Waveform"
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_category = "BB Waveform"
    draw = draw_ui


class BB_OT_upscale(bpy.types.Operator):
    bl_idname = "waveform.upscale"
    bl_label = "+"
    bl_description = "Increase resolution level"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        if s.resolution_level >= 8:
            self.report({'WARNING'}, 'Maximum resolution level reached')
            return {'CANCELLED'}
        
        s.resolution_level += 1
        s.resolution = 4000 * s.resolution_level
        
        self.report({'INFO'}, f'Resolution level: {s.resolution_level}')
        
        # Trigger rebuild with new resolution
        rebuild(context)
        return {'FINISHED'}


class BB_OT_downscale(bpy.types.Operator):
    bl_idname = "waveform.downscale"
    bl_label = "-"
    bl_description = "Decrease resolution level"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        if s.resolution_level <= 1:
            self.report({'WARNING'}, 'Minimum resolution level reached')
            return {'CANCELLED'}
        
        s.resolution_level -= 1
        s.resolution = 4000 * s.resolution_level
        
        self.report({'INFO'}, f'Resolution level: {s.resolution_level}')
        
        # Trigger rebuild with new resolution
        rebuild(context)
        return {'FINISHED'}


class BB_OT_all_channels(bpy.types.Operator):
    bl_idname = "waveform.all_channels"
    bl_label = "All Tracks"
    bl_description = "Enable all sequencer strips in red"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        if not context.scene.sequence_editor:
            self.report({'ERROR'}, 'No sequence editor')
            return {'CANCELLED'}
        
        seq = context.scene.sequence_editor
        strip_names = []
        
        # Get all sound strips
        for strip in seq.sequences:
            if strip.type == 'SOUND':
                strip_names.append(strip.name)
                
                # Initialize color entries with random colors if they don't exist
                # But DON'T modify existing colors
                found = False
                for item in s.strip_colors:
                    if item.name == strip.name:
                        found = True
                        break
                if not found:
                    new_item = s.strip_colors.add()
                    new_item.name = strip.name
                    new_item.color = get_random_color()
        
        if not strip_names:
            self.report({'ERROR'}, 'No sound strips found')
            return {'CANCELLED'}
        
        # Enable all strips and set red mode flag
        s.enabled_strips = ','.join(strip_names)
        s.all_channels_red = True  # Flag to display in red
        
        self.report({'INFO'}, f'Enabled {len(strip_names)} strips in red')
        rebuild(context)
        return {'FINISHED'}


class BB_OT_all_tracks(bpy.types.Operator):
    bl_idname = "waveform.all_tracks"
    bl_label = "All Tracks"
    bl_description = "Enable all audio tracks in red"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        s = context.scene.waveform_settings
        
        if not s.filepath:
            self.report({'ERROR'}, 'No file selected')
            return {'CANCELLED'}
        
        path = bpy.path.abspath(s.filepath)
        if not Path(path).exists():
            self.report({'ERROR'}, 'File does not exist')
            return {'CANCELLED'}
        
        # Get track count
        track_count = get_audio_track_count(path)
        
        if track_count == 0:
            self.report({'ERROR'}, 'No audio tracks found')
            return {'CANCELLED'}
        
        # Enable all tracks and initialize colors if needed
        for i in range(track_count):
            s.enabled_tracks[i] = True
            
            # Initialize color entries with random colors if they don't exist
            # But DON'T modify existing colors
            track_id = str(i)
            found = False
            for item in s.track_colors:
                if item.name == track_id:
                    found = True
                    break
            if not found:
                new_item = s.track_colors.add()
                new_item.name = track_id
                new_item.color = get_random_color()
        
        # Set the all_tracks_red flag to display in red
        s.all_tracks_red = True
        
        self.report({'INFO'}, f'Enabled {track_count} tracks in red')
        rebuild(context)
        return {'FINISHED'}


class BB_OT_select_channel(bpy.types.Operator):
    bl_idname = "waveform.select_channel"
    bl_label = "Select Tracks"
    bl_description = "Choose which sequencer strips to use for waveform"
    bl_options = {'REGISTER', 'UNDO'}
    
    def get_sound_strips(self, context):
        """Get all sound strips in the sequencer"""
        if not context.scene.sequence_editor:
            return []
        
        seq = context.scene.sequence_editor
        sound_strips = []
        for strip in seq.sequences:
            if strip.type == 'SOUND':
                sound_strips.append(strip)
        return sound_strips
    
    def invoke(self, context, event):
        sound_strips = self.get_sound_strips(context)
        
        if not sound_strips:
            self.report({'ERROR'}, 'No sound strips in sequencer')
            return {'CANCELLED'}
        
        # Initialize colors for strips that don't have them yet
        s = context.scene.waveform_settings
        
        # If enabled_strips is empty, enable only the first strip
        if not s.enabled_strips:
            s.enabled_strips = sound_strips[0].name
        
        # Only create color entries for new strips - preserve all existing colors
        # This MUST happen BEFORE clearing the all_channels_red flag
        for strip in sound_strips:
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
        
        return context.window_manager.invoke_popup(self, width=350)
    
    def draw(self, context):
        layout = self.layout
        sound_strips = self.get_sound_strips(context)
        s = context.scene.waveform_settings
        
        # Parse enabled strips
        enabled_names = set(s.enabled_strips.split(',')) if s.enabled_strips else set()
        
        layout.label(text="Select tracks to display:", icon='SOUND')
        layout.separator()
        
        col = layout.column(align=True)
        for strip in sound_strips:
            row = col.row(align=True)
            
            # Checkbox button
            is_enabled = strip.name in enabled_names
            op = row.operator("waveform.toggle_strip", text=f"Ch {strip.channel}: {strip.name}",
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
        enabled_names.discard('')  # Remove empty strings
        
        print(f"[STRIP TOGGLE DEBUG] Before toggle:")
        print(f"  enabled_strips string: '{s.enabled_strips}'")
        print(f"  enabled_names set: {enabled_names}")
        print(f"  enabled count: {len(enabled_names)}")
        print(f"  Toggling strip: '{self.strip_name}'")
        print(f"  Is this strip currently enabled? {self.strip_name in enabled_names}")
        
        # Check if trying to disable the last enabled strip
        is_currently_enabled = self.strip_name in enabled_names
        if is_currently_enabled and len(enabled_names) == 1:
            # Don't allow disabling the last strip
            self.report({'WARNING'}, 'At least one strip must be enabled')
            print(f"[STRIP TOGGLE] Cannot disable last strip")
            return {'CANCELLED'}
        
        # Toggle this strip
        if is_currently_enabled:
            enabled_names.remove(self.strip_name)
            print(f"[STRIP TOGGLE] Disabled: {self.strip_name}")
        else:
            enabled_names.add(self.strip_name)
            print(f"[STRIP TOGGLE] Enabled: {self.strip_name}")
        
        # Save back to property (filter out any empty strings)
        enabled_names.discard('')
        s.enabled_strips = ','.join(enabled_names) if enabled_names else ""
        
        print(f"[STRIP TOGGLE DEBUG] After toggle:")
        print(f"  enabled_strips string: '{s.enabled_strips}'")
        print(f"  enabled_names set: {enabled_names}")
        
        # Rebuild immediately
        rebuild(context)
        return {'FINISHED'}


class BB_OT_select_audio_track(bpy.types.Operator):
    bl_idname = "waveform.select_audio_track"
    bl_label = "Select Tracks"
    bl_description = "Choose which audio tracks to display"
    bl_options = {'REGISTER', 'UNDO'}
    
    def get_audio_tracks(self, context):
        """Get audio tracks from the current file"""
        s = context.scene.waveform_settings
        if s.source == "FILE":
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
        
        return context.window_manager.invoke_popup(self, width=350)
    
    def draw(self, context):
        layout = self.layout
        tracks = self.get_audio_tracks(context)
        s = context.scene.waveform_settings
        
        layout.label(text="Select tracks to display:", icon='SOUND')
        layout.separator()
        
        # Create rows with checkbox and color picker
        col = layout.column(align=True)
        for track in tracks:
            row = col.row(align=True)
            
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
        
        print(f"[TRACK TOGGLE] Track {self.track_index} now {'enabled' if s.enabled_tracks[self.track_index] else 'disabled'}")
        
        # Rebuild immediately
        rebuild(context)
        return {'FINISHED'}


classes = (BB_TrackColorItem, BB_StripColorItem, BB_WaveformSettings, 
           BB_PT_dopesheet, BB_PT_graph, 
           BB_OT_all_channels, BB_OT_all_tracks,
           BB_OT_select_channel, BB_OT_toggle_strip,
           BB_OT_select_audio_track, BB_OT_toggle_track)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.waveform_settings = bpy.props.PointerProperty(type=BB_WaveformSettings)


def unregister():
    global _color_update_timer
    
    # Cancel any pending timer
    if _color_update_timer is not None and _color_update_timer.is_alive:
        _color_update_timer.cancel()
    
    clear_handlers()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.waveform_settings
