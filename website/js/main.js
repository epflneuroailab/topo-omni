// Mobile navigation toggle
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

// Copy BibTeX to clipboard
const copyBtn = document.getElementById('copyBibtex');
const bibtexText = document.getElementById('bibtexText');

if (copyBtn && bibtexText) {
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(bibtexText.textContent.trim());
      copyBtn.textContent = 'Copied!';
      copyBtn.classList.add('copied');
      setTimeout(() => {
        copyBtn.textContent = 'Copy';
        copyBtn.classList.remove('copied');
      }, 1800);
    } catch (err) {
      copyBtn.textContent = 'Press Ctrl/Cmd+C';
    }
  });
}
