# Add Video Calls to Your App in an Afternoon

*Introducing the Digital Samba skill for Claude*

---

Video conferencing integration usually means weeks of work - wading through API docs, wrangling authentication, dealing with WebRTC headaches. There's a faster way.

The Digital Samba skill gives Claude everything it needs to build video apps with you. A few simple commands to install, and Claude learns our REST API, SDK methods, and common integration patterns. No configuration needed.

## See It in Action

[VIDEO EMBED]

We asked Claude to create an interview room app from scratch. It picked Next.js, set up API routes, and handled JWT authentication - all using patterns from the skill. A few minutes later, we had a working app.

Interviewers create a room, get a code to share. Candidates enter the code and join. The API handles the room, enables recording, generates tokens. Simple flow, no fuss.

The generated code follows best practices - secure JWT token generation, proper SDK initialization, and clean API integration patterns. You don't manage any video infrastructure - that's all on us.

## What's in the Skill

- **97 REST API endpoints** - Rooms, sessions, recordings, webhooks, and more
- **SDK methods and events** - Full reference for the embedded SDK
- **6 integration patterns** - Public rooms, authenticated users, webinars, recordings
- **JWT authentication guide** - Token generation in Node.js, Python, and PHP

## How to Install

**Claude Code (CLI)**

```bash
git clone https://github.com/digitalsamba/digital-samba-skill.git
cp -r digital-samba-skill/.claude/skills/digital-samba your-project/.claude/skills/
```

**Claude Desktop / Claude.ai**

1. Download the [latest release ZIP](https://github.com/digitalsamba/digital-samba-skill/releases/latest)
2. Go to Settings → Skills
3. Click "Add custom skill" and upload the ZIP

## Get Started

The Digital Samba skill is on GitHub. Go give it a try.

[View on GitHub](https://github.com/digitalsamba/digital-samba-skill) | [Digital Samba Documentation](https://developer.digitalsamba.com)

---

*Five minutes. No infrastructure to manage. Full control over the experience.*
