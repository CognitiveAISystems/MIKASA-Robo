document.addEventListener("DOMContentLoaded", () => {
  const blocks = document.querySelectorAll("div.highlight pre, pre.literal-block");

  blocks.forEach((pre) => {
    const container = pre.closest("div.highlight") || pre.parentElement;
    if (!container || container.querySelector(":scope > .mikasa-copy-button")) {
      return;
    }

    container.classList.add("mikasa-copy-container");

    const button = document.createElement("button");
    button.type = "button";
    button.className = "mikasa-copy-button";
    button.setAttribute("aria-label", "Copy code");
    button.textContent = "Copy";

    button.addEventListener("click", async () => {
      const text = pre.innerText.replace(/\n+$/, "");
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      } catch (error) {
        button.textContent = "Failed";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      }
    });

    container.insertBefore(button, container.firstChild);
  });
});
