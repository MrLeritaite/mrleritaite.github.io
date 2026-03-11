/**
 * Add listener for theme mode toggle
 */

const $toggle = document.getElementById('mode-toggle');
const $topbarToggle = document.getElementById('topbar-mode-toggle');

export function modeWatcher() {
  if ($toggle) {
    $toggle.addEventListener('click', () => {
      Theme.flip();
    });
  }

  if ($topbarToggle) {
    $topbarToggle.addEventListener('click', () => {
      Theme.flip();
    });
  }
}
