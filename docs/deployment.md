# jevrl.com deployment

The public site serves the four-game results, saved policy replays, animated
training gallery, and Key Quest. Training runs remain local: the public server
serves static files and does not expose the training API.

Host: `root@156.238.253.180:22`. Caddy serves
`/srv/jevrl.com/current`, a symlink to a versioned directory under
`/srv/jevrl.com/releases/`. The source is released under the MIT license.

Build from the project root:

```bash
uv run --extra classic python scripts/export_site.py --output dist/jevrl-com
uv run --extra classic python scripts/homepage_smoke.py --url https://jevrl.com/
tar -czf dist/jevrl-com.tar.gz -C dist/jevrl-com .
```

For a release, upload only the static archive, verify its SHA-256, extract it
into a new release directory, and atomically replace the `current` symlink.
Retain the previous release for rollback. Never upload the repository root,
credential files, or manuscript artifacts.

`deploy/Caddyfile.jevrl` contains this site's configuration. The server's
`/etc/caddy/Caddyfile` is shared by several services: append or update only the
JevRL block, keeping a backup, and validate before reloading Caddy. Update the
JevRL rows in `/srv/REGISTRY.md` and compare the server's
`/srv/zurv/bin/netcheck.sh` output with its pre-deployment baseline. Existing
unrelated warnings are not part of the JevRL deployment.

DNS is managed at Porkbun. Configure the apex A record to target
`156.238.253.180` and point `www` to the same host; the final Caddy configuration
redirects `www` to the apex. Caddy obtains and renews HTTPS
certificates automatically once DNS resolves to this server. Porkbun API
credentials are read from ignored local files and never copied into the site.

Verify HTTPS, the HTTP and www redirects, all four game replays, the gallery,
and the `/key-quest/` page after deploying. `/static/classic/paper.pdf`,
`/static/classic/jevrl-arxiv.tar.gz`, and `/api/classic/train` must return 404.
