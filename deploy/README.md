# Deploying the landing page

```
push to main (landing/ or deploy/) ─► build landing image ─► GHCR ─► ssh deploy@vps deploy.sh
                                                                        │
                        pull ─► up -d ─► healthy, reachable via Traefik? ─┬─ yes: done
                                                                          └─ no: previous tag back up, run fails
```

`landing/` is built by Vite and served by nginx (`deploy/landing.Dockerfile`).
Traefik runs beside it, installed and kept running by `deploy/ansible/`. It owns
80/443, gets Let's Encrypt certificates, and routes `SITE_DOMAIN` to the page.

## First-time setup

1. Point DNS for your domain at the VPS.
2. Create a deploy key: `ssh-keygen -t ed25519 -f ~/.ssh/logpipe-deploy -C logpipe-deploy`.
3. Provision the VPS (Docker, the `deploy` user, `/opt/logpipe-landing`, Traefik):
   ```sh
   cd deploy/ansible
   cp inventory.example.ini inventory.ini   # host, Let's Encrypt email, key path
   ansible-galaxy collection install -r requirements.yml
   ansible-playbook -i inventory.ini playbook.yml
   ```
   Re-run it any time; it starts Traefik only if Traefik isn't running or its
   config changed. On a box shared with other sites, keep
   `templates/traefik*.j2` identical to theirs: whichever playbook ran last wins.
4. In GitHub → Settings → Environments → `production`:

   | kind | name | value |
   |---|---|---|
   | secret | `VPS_SSH_KEY` | contents of `~/.ssh/logpipe-deploy` |
   | secret | `VPS_KNOWN_HOSTS` | `ssh-keyscan -p 22 <host>` |
   | secret | `PROD_ENV_FILE` | `deploy/.env.example`, filled in |
   | var | `VPS_HOST` | host or IP |
   | var | `VPS_PORT` | optional, default 22 |

5. Push to `main`, or run **Deploy landing** by hand.

## Day to day

- **Release**: merge a change under `landing/` to `main`.
- **Roll back**: Actions → Deploy landing → Run workflow → `image_tag` = an earlier commit SHA.
- **Change a setting**: edit `PROD_ENV_FILE`, re-run the latest Deploy landing.
- **Logs**: `ssh deploy@vps 'cd /opt/logpipe-landing && docker compose -f docker-compose.prod.yml logs -f web'`.
- **Build locally**: `docker build -f deploy/landing.Dockerfile -t logpipe-landing .`
