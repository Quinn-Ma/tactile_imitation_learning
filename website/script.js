"use strict";
document.getElementById("copy-citation").addEventListener("click", async () => {
  const citation = document.getElementById("bibtex");
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = "BibTeX copied.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Press Ctrl+C or Command+C to copy.";
  }
});
