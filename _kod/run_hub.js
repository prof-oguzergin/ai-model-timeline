// index.html ana script'ini sahte DOM ile calistirip CALISMA-ANI hatasini yakalar.
// node --check yalnizca sozdizimini gorur; tanimsiz degisken gibi kapsam
// hatalarini yakalamaz. (Bir kez canli sayfada UC bolumu birden dusurmustu.)
// Kullanim: node run_hub.js <index.html yolu>
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const code = blocks.reduce((a, b) => (b.length > a.length ? b : a), '');

function sahteOge() {
  const o = {
    style: {}, dataset: {}, cells: [], rows: [], children: [], childNodes: [],
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    textContent: '', innerHTML: '', innerText: '', value: '', href: '', src: '',
    hidden: false, offsetWidth: 800, offsetHeight: 600, clientWidth: 800,
    getContext: () => ({}), addEventListener(){}, removeEventListener(){},
    appendChild(){}, removeChild(){}, insertBefore(){}, setAttribute(){},
    getAttribute(){ return null; }, remove(){}, focus(){}, click(){},
    querySelector: () => sahteOge(), querySelectorAll: () => [],
    getBoundingClientRect: () => ({top:0,left:0,width:800,height:600,bottom:600,right:800}),
    scrollIntoView(){},
  };
  return o;
}
const belge = {
  getElementById: () => sahteOge(),
  querySelector: () => sahteOge(),
  querySelectorAll: () => [],
  createElement: () => sahteOge(),
  addEventListener(){}, body: sahteOge(), documentElement: sahteOge(),
  head: sahteOge(), cookie: '', title: '', readyState: 'complete',
};
const g = globalThis;
g.document = belge;
g.window = g;
g.location = { href: 'https://yapayzeka.oguzergin.net/', search: '', hash: '', pathname: '/' };
g.navigator = { language: 'tr', userAgent: 'node' };
g.localStorage = { getItem: () => null, setItem(){}, removeItem(){} };
g.IntersectionObserver = class { constructor(){} observe(){} unobserve(){} disconnect(){} };
g.ResizeObserver = class { constructor(){} observe(){} unobserve(){} disconnect(){} };
g.MutationObserver = class { constructor(){} observe(){} disconnect(){} };
g.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve('') });
g.matchMedia = () => ({ matches: false, addEventListener(){}, addListener(){} });
g.requestAnimationFrame = (fn) => { try { fn(0); } catch (e) { console.log('rAF HATA:', e.message); } };
g.addEventListener = () => {};
g.scrollTo = () => {};

try {
  new Function(code)();
  console.log('SONUC: hatasiz calisti');
} catch (e) {
  console.log('CALISMA-ANI HATASI:', e.message);
  console.log(e.stack.split('\n').slice(0, 4).join('\n'));
  process.exitCode = 1;
}
