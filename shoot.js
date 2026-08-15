// Screenshot every dashboard tab and report console errors.
const puppeteer = require('puppeteer');

const URL = process.argv[2] || 'http://localhost:8788';
const TABS = ['video', 'prevalence', 'episodes', 'validation', 'corpus', 'method'];

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1100 });

  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('requestfailed', r =>
    errors.push('requestfailed: ' + r.url() + ' ' + (r.failure() || {}).errorText));

  await page.goto(URL, { waitUntil: 'networkidle2', timeout: 60000 });
  await page.waitForFunction("document.querySelectorAll('#kpis .card').length >= 5",
                             { timeout: 20000 });

  for (const tab of TABS) {
    await page.evaluate(t => document.querySelector(`.tabbtn[data-k="${t}"]`).click(), tab);
    await new Promise(r => setTimeout(r, 900));
    await page.screenshot({ path: `shots/${tab}.png`, fullPage: true });

    const txt = await page.evaluate(() => document.getElementById('main').innerText);
    const junk = ['undefined', 'NaN', 'Infinity', '[object Object]']
      .filter(j => txt.includes(j));
    console.log(`${tab.padEnd(12)} ${String(txt.length).padStart(6)} chars` +
                (junk.length ? `   JUNK: ${junk.join(', ')}` : ''));
  }

  // video tab specifics
  await page.evaluate(() => document.querySelector('.tabbtn[data-k="video"]').click());
  await new Promise(r => setTimeout(r, 1200));
  const v = await page.evaluate(async () => {
    const vid = document.getElementById('vid');
    const strip = document.getElementById('strip');
    await new Promise(r => {
      if (vid.readyState >= 2) return r();
      vid.addEventListener('loadeddata', r, { once: true });
      setTimeout(r, 8000);
    });
    vid.currentTime = 40;
    await new Promise(r => setTimeout(r, 600));
    return {
      videoReady: vid.readyState,
      duration: Math.round(vid.duration * 10) / 10,
      videoW: vid.videoWidth, videoH: vid.videoHeight,
      pathPts: (strip.querySelector('path')?.getAttribute('d') || '').length,
      bands: strip.querySelectorAll('rect').length,
      segRows: document.querySelectorAll('#seglist .seg').length,
      clipButtons: document.querySelectorAll('.clipbtn').length,
      readout: document.getElementById('readout').innerText,
      playhead: document.getElementById('ph').getAttribute('x1'),
    };
  });
  console.log('\nvideo tab:', JSON.stringify(v, null, 2));
  await page.screenshot({ path: 'shots/video_seeked.png', fullPage: true });

  // filter box
  await page.evaluate(() => document.querySelector('.tabbtn[data-k="episodes"]').click());
  await new Promise(r => setTimeout(r, 500));
  const before = await page.evaluate(() => document.querySelectorAll('#rows tr').length);
  await page.type('#q', 'fold');
  await new Promise(r => setTimeout(r, 400));
  const after = await page.evaluate(() => document.querySelectorAll('#rows tr').length);
  console.log(`filter: ${before} rows -> ${after} rows for "fold"`);

  console.log('\n' + (errors.length ? 'ERRORS:\n  ' + errors.join('\n  ')
                                    : 'no console errors, no failed requests'));
  await browser.close();
})();
