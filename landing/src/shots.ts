import type { Shot } from './components/Screenshot'

// Native sizes of the files in public/screenshots; width/height reserve the space before they load.
export const SHOTS = {
  explorer: {
    src: '/screenshots/log-explorer.webp', width: 1878, height: 985,
    alt: "logpipe's Log Explorer: a live chart of log volume over a table of parsed nginx, postgres and shop-api lines, with facets for status, service, host and source.",
  },
  templates: {
    src: '/screenshots/templates.webp', width: 1880, height: 992,
    alt: "logpipe's Templates page: every log shape the parser knows, with fields such as client_ip, timestamp and duration_ms highlighted, line counts, and new shapes marked unseen.",
  },
  incident: {
    src: '/screenshots/incident.webp', width: 1872, height: 984,
    alt: 'An incident on shop-api: payment warnings and errors highlighted as kinds of lines the model never learned, 279 of 400 lines unusual.',
  },
  rootCause: {
    src: '/screenshots/root-cause.webp', width: 1862, height: 991,
    alt: "logpipe's root cause answer: PostgreSQL ran out of connection slots, citing log lines 18 and 19, with a timeline and next steps.",
  },
  ranking: {
    src: '/screenshots/ranking.webp', width: 1862, height: 990,
    alt: 'The ranking behind the answer: each template scored by anomaly, rarity and proximity, the postgres FATAL line first.',
  },
  ask: {
    src: '/screenshots/ask.webp', width: 1860, height: 990,
    alt: "The Query page's Ask tab: the question 'why did the shop break in the last 15 minutes?' answered with likely causes, a timeline and next steps.",
  },
} satisfies Record<string, Shot>
