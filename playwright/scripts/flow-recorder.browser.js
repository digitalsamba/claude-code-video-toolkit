// Loaded as plain browser JavaScript, avoiding transpiler helpers in serialized
// Playwright init scripts. `masks` and the three bindings come from flow.ts.
let pending = Promise.resolve();
window.__toolkitFlush = () => pending;

function selector(element) {
  const testId = element.getAttribute('data-testid');
  if (testId) return `[data-testid=${JSON.stringify(testId)}]`;
  if (element.id) return `#${CSS.escape(element.id)}`;
  const parts = [];
  let node = element;
  while (node && node !== document.documentElement) {
    const tag = node.tagName.toLowerCase();
    const parent = node.parentElement;
    const siblings = parent ? Array.from(parent.children).filter(child => child.tagName === node.tagName) : [];
    parts.unshift(`${tag}:nth-of-type(${siblings.indexOf(node) + 1})`);
    node = parent;
  }
  return `html > ${parts.join(' > ')}`;
}

function send(step) {
  pending = pending.then(async () => {
    await window.__toolkitRecordStep(typeof step === 'function' ? await step() : step);
  });
}

document.addEventListener('click', event => {
  if (!(event.target instanceof Element)) return;
  const target = event.target.closest('button,a,input,textarea,select,[role="button"]') || event.target;
  send({kind: 'click', selector: selector(target)});
}, true);

document.addEventListener('input', event => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) return;
  if (target instanceof HTMLInputElement && ['checkbox', 'radio', 'file'].includes(target.type)) return;
  const selected = selector(target);
  const hidden = (target instanceof HTMLInputElement && target.type === 'password') || masks.some(mask => target.matches(mask));
  const value = target.value;
  if (hidden) send(async () => ({kind: 'fill', selector: selected,
    value: {env: await window.__toolkitSecretName(selected)}}));
  else send({kind: 'fill', selector: selected, value});
}, true);

document.addEventListener('change', event => {
  if (event.target instanceof HTMLSelectElement) {
    send({kind: 'select', selector: selector(event.target), value: event.target.value});
  }
}, true);

document.addEventListener('keydown', event => {
  if (event.target instanceof Element && ['Enter', 'Tab', 'Escape'].includes(event.key)) {
    send({kind: 'press', selector: selector(event.target), key: event.key});
  }
}, true);
