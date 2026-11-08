// Entrances, the hero shutter, the stage slider and the copy button. Without this file every word and picture
// is still on the page: motion is added on top, never needed.
document.documentElement.classList.add('js');

const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// staggered entrances, once per element
const io = 'IntersectionObserver' in window && new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); }
}, { rootMargin: '0px 0px -8% 0px' });
document.querySelectorAll('[data-rise], .shutter').forEach((el) => (io ? io.observe(el) : el.classList.add('is-in')));
document.querySelectorAll('[data-stagger]').forEach((group) => {
  [...group.querySelectorAll('[data-rise]')].forEach((el, i) => el.style.setProperty('--i', i));
});

// the hero loop pauses when off screen or under reduced motion
document.querySelectorAll('video[data-loop]').forEach((v) => {
  if (reduce) { v.removeAttribute('autoplay'); v.pause(); return; }
  const vo = 'IntersectionObserver' in window && new IntersectionObserver(([e]) => (e.isIntersecting ? v.play().catch(() => {}) : v.pause()));
  if (vo) vo.observe(v);
});

// drag through the stages
const stages = document.querySelector('[data-stages]');
if (stages) {
  const data = JSON.parse(stages.querySelector('script[type="application/json"]').textContent);
  const input = stages.querySelector('input[type="range"]');
  const clause = stages.querySelector('[data-clause]');
  const cost = stages.querySelector('[data-cost]');
  const cmd = stages.querySelector('[data-cmd]');
  const out = stages.querySelector('[data-out]');
  const term = stages.querySelector('.proof-in');
  const img = stages.querySelector('[data-img]');
  const buttons = [...stages.querySelectorAll('ol button')];
  let shown = -1;
  const show = (i) => {
    if (i === shown) return;
    shown = i;
    const s = data[i];
    input.value = i;
    input.style.setProperty('--fill', `${(i / (data.length - 1)) * 100}%`);
    input.setAttribute('aria-valuetext', s.name);
    buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
    const swap = () => {
      clause.textContent = s.clause;
      cost.innerHTML = s.cost;
      cmd.textContent = s.cmd;
      out.innerHTML = s.out;
      img.src = s.img;
      img.alt = s.alt;
      clause.style.opacity = term.style.opacity = 1;
    };
    if (reduce) return swap();
    clause.style.opacity = term.style.opacity = 0;
    setTimeout(swap, 200);
  };
  input.addEventListener('input', () => show(Number(input.value)));
  buttons.forEach((b, j) => b.addEventListener('click', () => show(j)));
  data.forEach((s) => { const pre = new Image(); pre.src = s.img; });
  show(0);
}

// copy a command
document.querySelectorAll('[data-copy]').forEach((b) => {
  b.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(b.dataset.copy);
      const was = b.textContent;
      b.textContent = 'Copied';
      setTimeout(() => (b.textContent = was), 1600);
    } catch { /* the command is on the page to select by hand */ }
  });
});
