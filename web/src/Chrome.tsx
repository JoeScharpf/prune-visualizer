function YCSquare() {
  return (
    <a
      href="https://www.ycombinator.com/companies/overshoot"
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Backed by Y Combinator"
      className="yc-square site-yc inline-flex items-center justify-center"
      style={{ width: 32, height: 32 }}
    >
      <svg
        width="32"
        height="32"
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path className="yc-square-bg" d="M0 0H40V40H0V0Z" />
        <path
          d="M18.6675 22.6661L11.7358 9.73291H14.9362L18.9362 17.8665C18.9362 18 19.0697 18.1335 19.2032 18.2671C19.3368 18.4006 19.3368 18.5342 19.4703 18.7997L19.6039 18.9332V19.0668C19.7374 19.3339 19.7374 19.4674 19.871 19.7345C20.0045 19.868 20.0045 20.1351 20.1381 20.2671C20.2716 19.8665 20.5387 19.5994 20.6707 19.0668C20.8042 18.6661 21.0713 18.2671 21.3384 17.8665L25.3384 9.73291H28.2716L21.34 22.8013V31.0684H18.6738L18.6675 22.6661Z"
          fill="#FFFFFF"
        />
      </svg>
    </a>
  );
}

export function Nav() {
  // Absolute overshoot.ai URLs — this app is hosted on its own domain, so
  // relative /blogs, /join, etc. would 404 here.
  const links = [
    { label: "Platform", href: "https://platform.overshoot.ai" },
    { label: "Documentation", href: "https://docs.overshoot.ai" },
    { label: "Blog", href: "https://overshoot.ai/blogs" },
    { label: "Join us", href: "https://overshoot.ai/join" },
  ];
  return (
    <header
      className="fixed top-0 inset-x-0 z-50 w-full h-[60px]"
      style={{ backgroundColor: "#FAFAF9" }}
    >
      <div className="container-1312 h-full flex items-center justify-between">
        <a
          href="https://overshoot.ai/"
          className="flex items-center"
          aria-label="Overshoot"
        >
          <img
            src="/overshoot-logo.svg"
            alt="Overshoot"
            width={125}
            height={20}
            className="block"
          />
        </a>

        <nav className="site-nav-desktop flex items-center" aria-label="Main">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              className="btn-slide h-9 inline-flex items-center px-4 py-2 text-sm font-medium text-fg"
            >
              <span
                aria-hidden
                className="btn-slide-fill"
                style={{ background: "#F3EFED" }}
              />
              <span className="btn-slide-content">{l.label}</span>
            </a>
          ))}
        </nav>

        <YCSquare />

        {/* Mobile: CSS-only disclosure menu (no JS; closes on navigation). */}
        <details className="site-nav-menu">
          <summary className="site-nav-toggle" aria-label="Menu">
            <span className="site-nav-bars" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path
                  d="M3 6h14M3 10h14M3 14h14"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </span>
            <span className="site-nav-x" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path
                  d="M5 5l10 10M15 5L5 15"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </span>
          </summary>
          <div className="site-nav-drawer">
            {links.map((l) => (
              <a key={l.label} href={l.href}>
                {l.label}
              </a>
            ))}
          </div>
        </details>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer
      className="w-full border-t border-stone-200 mt-auto"
      style={{ backgroundColor: "#FAFAF9", color: "#78716C" }}
    >
      <div className="container-1312 site-footer-row flex items-center justify-between">
        <a
          href="https://platform.overshoot.ai/privacy.html"
          className="hover:text-stone-950 transition-colors"
        >
          Privacy
        </a>
        <a
          href="https://overshoot.ai/faq"
          className="hover:text-stone-950 transition-colors"
        >
          FAQ
        </a>
        <a
          href="https://overshoot.ai/models"
          className="hover:text-stone-950 transition-colors"
        >
          Models
        </a>
        <a
          href="https://overshoot.ai/usages"
          className="hover:text-stone-950 transition-colors"
        >
          Usages
        </a>
        <a
          href="mailto:founders@overshoot.ai"
          className="hover:text-stone-950 transition-colors"
        >
          founders@overshoot.ai
        </a>
      </div>
    </footer>
  );
}
