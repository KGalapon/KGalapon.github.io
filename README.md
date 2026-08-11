# Karl Galapon — Personal Blog

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy?color=brightgreen)](https://rubygems.org/gems/jekyll-theme-chirpy)
[![Build and Deploy](https://github.com/KGalapon/KGalapon.github.io/actions/workflows/pages-deploy.yml/badge.svg)](https://github.com/KGalapon/KGalapon.github.io/actions/workflows/pages-deploy.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

My personal blog for data science, statistics, and mathematics — built with [Jekyll](https://jekyllrb.com/) and the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme, hosted on GitHub Pages.

🔗 **Live site:** [kgalapon.github.io](https://KGalapon.github.io)

## About

I build tools that turn messy, real-world data into something a person can actually act on. As a 3rd-year Applied Mathematics – Data Science student at Ateneo de Manila University, most of what I do starts with data that doesn't exist yet — scraping job postings, pulling records into a warehouse, cleaning text that was never meant to be structured. This means I spend as much time on the pipeline as on the model (e.g., writing a scraper for a specific site's HTML, or using an LLM to pull years-of-experience out of a free-text job description), because a good model on bad data is still a bad result. However, the pipeline is only half the job; the other half is putting what it found in front of someone, which is why most of my projects end as a dashboard rather than a notebook. This blog documents that full path — scraper to dashboard — one project at a time.

## Built With

- [Jekyll](https://jekyllrb.com/) — static site generator
- [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy) (via [chirpy-starter](https://github.com/cotes2020/chirpy-starter))
- Ruby / Bundler
- GitHub Actions for CI/CD, deployed automatically to GitHub Pages

## Getting Started

### Prerequisites

- Ruby (see the [Jekyll installation guide](https://jekyllrb.com/docs/installation/) for your OS)
- Bundler (`gem install bundler`)
- Git

### Local Setup

1. Clone the repo and its submodules (the theme pulls static assets from a submodule):

   ```bash
   git clone --recurse-submodules https://github.com/KGalapon/KGalapon.github.io.git
   cd KGalapon.github.io
   ```

2. Install dependencies:

   ```bash
   bundle install
   ```

3. Run the site locally:

   ```bash
   bundle exec jekyll serve
   ```

4. Open [http://127.0.0.1:4000](http://127.0.0.1:4000) in your browser.

> **Tip:** A [Dev Container](.devcontainer) config is included, so you can also open this repo in VS Code / GitHub Codespaces and get a ready-to-go Jekyll environment automatically.

## Project Structure

```
.
├── _config.yml       # Site configuration (title, author, social links, etc.)
├── _data             # Site data files
├── _includes         # Reusable HTML/Liquid partials
├── _plugins          # Custom Jekyll plugins
├── _posts            # Blog posts
├── _tabs             # Static pages (About, Archives, Categories, Tags)
├── assets            # Images, styles, and other static assets
├── tools             # Helper scripts (e.g. for creating new posts)
└── index.html        # Homepage entry point
```

## Writing a New Post

Create a new Markdown file in `_posts/` following the naming convention:

```
YYYY-MM-DD-title-of-post.md
```

with front matter like:

```yaml
---
title: My Post Title
date: 2026-01-01 12:00:00 +0800
categories: [Category, Subcategory]
tags: [tag1, tag2]
---
```

See the [Chirpy theme docs](https://github.com/cotes2020/jekyll-theme-chirpy/wiki) for the full list of supported front matter options (pinning posts, table of contents, math rendering, images, etc.).

## Deployment

This site deploys automatically via the GitHub Actions workflow in [`.github/workflows/pages-deploy.yml`](.github/workflows/pages-deploy.yml) whenever changes are pushed to `main`. No manual build step is needed — just commit and push.

## License

This project is published under the [MIT License](LICENSE). The underlying theme, [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy/), is also MIT licensed.

## Contact

**Karl Chester Galapon**
📧 [karlchestergalapon77@gmail.com](mailto:karlchestergalapon77@gmail.com)
🐙 [GitHub](https://github.com/KGalapon) · 📘 [Facebook](https://www.facebook.com/karlchester.galapon.9/)
