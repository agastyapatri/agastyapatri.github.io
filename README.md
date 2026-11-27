# agastyapatri.com

Personal site and blog, built with [Hugo](https://gohugo.io) and a small hand-written theme (no external theme, no JavaScript).

## Write

```sh
hugo new posts/my-post.md     # creates a draft with title, date, description, tags
hugo server -D                # preview at http://localhost:1313, drafts included
```

Set `draft = false` and push to `main` to publish. The `description` line shows under the title on the homepage, so write one for every post.

## Where things live

| Path | What it is |
| --- | --- |
| `content/_index.md` | Homepage intro |
| `content/posts/` | Blog posts |
| `content/projects/` | Projects list (each entry links out via `link`) |
| `hugo.toml` | Site title, tagline, menu, social links |
| `assets/css/main.css` | The whole look. Colours and fonts are the variables at the top |
| `layouts/` | Page templates |
| `static/` | Files served as-is: fonts, favicon, profile photo |

## Deploy

Cloudflare Pages builds `main` on every push: build command `hugo --minify`, output `public`, environment variable `HUGO_VERSION=0.170.1`.

`.github/workflows/deploy.yml` still publishes agastyapatri.github.io. After agastyapatri.com is live, set `REDIRECT_TO_NEW_DOMAIN: "true"` in that file so old links redirect to the new domain.

Fonts: [Libron](https://github.com/nicoverbruggen/libron) (v0.32, built from source) and JetBrains Mono, both under the SIL Open Font License (see `static/fonts/`).
