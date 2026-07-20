export function Nav() {
  return (
    <header
      className="fixed top-0 inset-x-0 z-50 w-full h-[60px]"
      style={{ backgroundColor: "#FAFAF9" }}
    >
      <div className="container-1312 h-full flex items-center justify-between">
        <a href="/" className="flex items-center" aria-label="Overshoot">
          <img
            src="/overshoot-logo.svg"
            alt="Overshoot"
            width={125}
            height={20}
            className="block"
          />
        </a>
        <span className="demo-label text-fg-muted">Token pruning visualizer</span>
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
        <span>HiPrune / HyDART token pruning visualizer</span>
        <span className="font-mono">Overshoot</span>
      </div>
    </footer>
  );
}
