# Recommended Parameter Values for Hunyuan3D 2.1 Texture Pipeline

## High Quality Defaults (for any input)

| Parameter | Recommended | Default | Why |
|-----------|-------------|---------|-----|
| steps | 20-30 | 10 | More denoising steps = cleaner textures with less noise |
| guidance_scale | 3.0-4.0 | 3.0 | Default is good; above 5.0 risks overexposure |
| texture_size | Match upscaled view resolution (e.g. 2048 if using 4x upscaler) | 1024 | Avoids wasting upscale detail during baking |
| render_size | 1024 | 1024 | Default is fine; increase to 2048 only for very high-res output |
| bake_exp | 3 | 4 | Slightly more even blending across views, fewer visible seams |
| bake_angle_thres | 70 | 75 | Accepts slightly more data from oblique angles, better coverage |
| inpaint_radius | 5 for 1024 textures, 10 for 2048 | 3 | Default is too small; larger radius fills UV seam gaps properly |
| inpaint_method | NS | NS | Navier-Stokes is better for the typical gap sizes in UV baking |
| scheduler | UniPCMultistep | EulerAncestralDiscrete | Better quality at low step counts (10-20) |
| guidance_rescale | 0.0-0.3 | 0.0 | Use 0.3 only if you see overexposed/washed-out areas |
| camera_distance | 1.1 | 1.1 | Default works for most meshes |
| mesh_scale_factor | 1.15 | 1.15 | Default works for most meshes |

## Per Use-Case Tuning

### Organic/Round objects (characters, animals, plush)
- `bake_angle_thres`: 65-70 (round surfaces need more oblique coverage)
- `bake_exp`: 3 (smoother blending across curved surfaces)
- Camera config: 8+ views for full coverage of curves

### Hard-surface/Mechanical (furniture, vehicles, architecture)
- `bake_angle_thres`: 75-80 (flat surfaces, oblique data is low quality)
- `bake_exp`: 4-5 (prefer head-on views, sharper results)
- Camera config: 6 views with heavier cardinal weights

### High-res production output
- Use 4x upscaler on multiviews
- Set `texture_size` = 2048 or 4096 to match
- Set `inpaint_radius` = 10-15
- Set `steps` = 25-30

## Camera Config Recommendations (8-view setup)

```
azimuths:    0, 90, 180, 270, 0, 180, 45, 315
elevations:  0, 0, 0, 0, 90, -90, 0, 0
weights:     1, 0.5, 1, 0.5, 1, 1, 0.1, 0.1
ortho_scale: 1.1
```

This provides full 360 degree + top/bottom coverage with diagonal fill views.
