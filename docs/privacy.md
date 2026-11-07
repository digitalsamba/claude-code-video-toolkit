# Privacy policy

This covers claude-code-video-toolkit and its YouTube uploader (`tools/youtube_upload.py`, the `/publish`
command). Last updated 2026-11-07.

## What it is

An open-source toolkit you run on your own computer. There is no hosted service, no account with us and no
server of ours in between: the code talks directly to the services you configure (YouTube, and any AI providers
you set up yourself).

## YouTube data

When you log in with `youtube_upload.py --auth`, Google asks you to let the uploader manage your YouTube videos.
With that permission it:

- uploads the video files you choose, with the title, description, tags, thumbnail, captions and privacy setting
  you give it;
- reads your channel's name and upload status, to report what happened.

It does not read, change or delete any other videos, comments, subscribers or account data.

## Where your data lives

- The OAuth token Google returns is saved only on your computer, in `_internal/.youtube/` inside your copy of the
  toolkit (that folder is excluded from git). Delete the folder to remove it, and revoke access at any time at
  https://myaccount.google.com/permissions.
- Videos and their metadata go from your computer to YouTube only.
- Nothing is sent to the toolkit's authors, and the toolkit collects no analytics or telemetry.

Use of information received from Google APIs follows the
[Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including the Limited Use requirements. YouTube's own terms apply to what you upload:
https://www.youtube.com/t/terms and https://policies.google.com/privacy.

## Other services

Other tools in the toolkit send your prompts, audio or images to the AI providers you configure (for example
Modal, RunPod, ElevenLabs, Ideogram), under those providers' own terms. Each is opt-in and uses your own API key.

## Contact

Open an issue at https://github.com/digitalsamba/claude-code-video-toolkit/issues.
