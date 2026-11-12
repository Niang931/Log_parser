// The logpipe mark (docs/assets/logo.svg): raw lines in, a pipe, structured fields out.
export function LogoMark({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 128 128" aria-hidden="true">
      <rect width="128" height="128" rx="28" fill="#0f2a3d" />
      <rect x="14" y="40" width="30" height="7" rx="3.5" fill="#7fb7d9" />
      <rect x="14" y="60.5" width="22" height="7" rx="3.5" fill="#7fb7d9" opacity=".75" />
      <rect x="14" y="81" width="34" height="7" rx="3.5" fill="#7fb7d9" opacity=".5" />
      <path d="M48 34h24a8 8 0 0 1 8 8v44a8 8 0 0 1-8 8H48a8 8 0 0 1-8-8V42a8 8 0 0 1 8-8z" fill="#1e88e5" />
      <rect x="86" y="40" width="10" height="7" rx="2" fill="#ffb74d" />
      <rect x="99" y="40" width="16" height="7" rx="2" fill="#fff" />
      <rect x="86" y="60.5" width="10" height="7" rx="2" fill="#ffb74d" />
      <rect x="99" y="60.5" width="16" height="7" rx="2" fill="#fff" />
      <rect x="86" y="81" width="10" height="7" rx="2" fill="#ef5350" />
      <rect x="99" y="81" width="16" height="7" rx="2" fill="#fff" />
    </svg>
  );
}
