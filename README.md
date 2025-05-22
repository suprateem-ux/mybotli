# Overview

**BotLi** is a bot for Lichess. It connects any [UCI](https://backscattering.de/chess/uci/) engine with the [Lichess Bot API](https://lichess.org/api#tag/Bot).

It has a customizable support of Polyglot opening books, a variety of supported online opening books and an online endgame tablebase. It can query local Syzygy and Gaviota endgame tablebases.

In addition, BotLi can autonomously challenge other bots in any variant. It supports custom opening books and engines depending on color, time control and Lichess chess variant.

If you have found a bug, please [create an issue](https://github.com/Torom/BotLi/issues/new?labels=bug&template=bug_report.md). For discussion, feature requests and help join the [BotLi Discord server](https://discord.gg/6aS945KMFD).

# How to install

- **NOTE: Only Python 3.11 or later is supported!**
- Download the repo into BotLi directory: `git clone https://github.com/Torom/BotLi.git`
- Navigate to the directory in cmd/Terminal: `cd BotLi`
- Copy `config.yml.default` to `config.yml`

Install all requirements:
```bash
python -m pip install -r requirements.txt
