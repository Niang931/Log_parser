import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import './page.css'
import App from './App.tsx'

// The kit scripts in index.html (system.js, motion.js, layout.js) run right after this module and
// read the DOM once, so the page must exist before they start: render synchronously.
const root = createRoot(document.getElementById('root')!)
flushSync(() => root.render(<App />))
