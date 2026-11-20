# YouTube channel art: banner, avatar and video watermark

What a channel needs, YouTube's sizes, and what making a set for a real channel taught us (Ireland in the
Telling, 2026-11). Check every piece at the size people will see it before anyone reviews it.

## The three pieces

| Piece | Upload | Shown | Notes |
|-------|--------|-------|-------|
| Banner | 2560x1440, under 6 MB | desktop: the full-width band 2560x423 (y 508 to 931); phones: only the centre 1546x423 (x 507 to 2053); TV: all of it | Everything that matters goes in the centre 1546x423 |
| Avatar | 800x800 square | a circle, down to about 24 px beside comments | One shape on a plain ground; check 98, 48 and 24 px on light and dark |
| Video watermark | square, at least 150x150, under 1 MB, PNG with transparency | bottom right of every video, about 60 px on a 1280x720 player, dimmed until the viewer hovers | Make it 300x300 so it stays sharp on large players |

The API can set the banner and the watermark (`channelBanners.insert` then `channels.update` with
`brandingSettings`, and `watermarks.set`, all under the `youtube` scope `youtube_upload.py` already asks for).
It cannot set the avatar: that one is always uploaded by hand in YouTube Studio.

## Previews to make every time

- **Banner:** the full image with the desktop band and the phone area outlined, the desktop crop, the phone
  crop enlarged, and the phone crop at real size (390 px wide, about 107 px tall). Put the planned title on
  the crops, even a placeholder, to judge the space around it.
- **Avatar:** the circle at 300 px, then 98, 48 and 24 px on a light ground and a dark ground.
- **Watermark:** over light, dark and mid-grey footage, at about 60 px on a 1280x720 frame, at about 60%
  opacity (as shown until hover) and at full strength, with the corner enlarged.

## Lessons

- **Measure the clear space around the title; don't eyeball it.** On the phone crop, find the darkest ink in
  the title's rows either side of it. A strand of hair that looked clear was touching the first letter. About
  100 px of clear space each side of the title (in banner pixels) read well. Changing the title font or size
  changes it, so measure again after any type change.
- **FLUX puts the subject where it likes, which is usually below the band.** In six banners asked for a
  subject "at the vertical middle", FLUX put it below the visible band every time, so phones showed only sky.
  Ideogram's bounding boxes (`bbox`, y first, 0 to 1000) put it in the band every time. For banners, use
  Ideogram for layout, or compose the banner from layers yourself.
- **Ideogram adds things to painterly scenes that nobody asked for:** fake signatures in a corner, a red
  seal stamp, small figures on a far bank, a dancing group beside a monument, a bird. Ban them in the
  background and element descriptions ("no people, no birds, no signature, no seal, no text"), and still
  inspect the full image. A corner signature sits outside the desktop band but shows on TV.
- **Build banners from layers when the faces or figures may change.** A background plate, plus each
  portrait blended in with a soft oval mask, lets someone's own drawing replace a generated face later
  without regenerating the rest. Blending the portrait's edge with multiply makes a drawing's white paper
  disappear into the scene while the face keeps its own colour. Force the mask to zero at the square's edge,
  or the edge of the portrait shows as a line.
- **A mist veil behind the title** (a soft ellipse of the scene's own colour, sampled and lifted) gives the
  title room without changing the sky above it. Apply it before the faces so they stay crisp.
- **The avatar is the watermark.** A disc version of the avatar read on every kind of footage, dimmed too. A
  pale silhouette with a thin dark edge nearly disappeared on pale footage while dimmed, which is how most
  viewers see it.
- **A silhouette traced from a painting** works as an avatar when it's simplified. Stone separates from a
  painted sky and grass by low saturation and a blue-grey hue; Pillow's Min/Max filters and a flood fill from
  a point on the object keep just that shape. Traced as painted, two standing stones merged into one block;
  drawn apart, with the ground showing between them, the shape still read at 24 px.
- **Lighting words decide colour.** "Cool light across one cheek" came back near-greyscale while "ember
  light" came back warm, so two portraits in one banner looked mismatched. Fix it locally by giving the cool
  face the warm face's skin colour (YCbCr chroma, masked to the face, a little cooler) and leaving the hair
  and clothes alone.
- **Reference images from search are other people's work.** Don't crop them into channel art. Describe what
  the reviewer likes about them (a braided grey beard, a knotwork headband, a green cloak and a bronze
  brooch) and generate from the words.
- **Convert review voice notes locally:** `ffmpeg` to 16 kHz WAV, then Whisper medium twice (with and
  without a vocabulary prompt). Mark words both passes heard oddly ("Gare's grand" was "she's grand")
  rather than guessing silently.
