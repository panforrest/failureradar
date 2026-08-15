// Drive the dashboard and capture frames, one shot per narration segment.
// Frames are rendered deterministically (we set video.currentTime ourselves)
// so playback is smooth and each shot matches its audio length exactly.
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const puppeteer = require('puppeteer');
const ffmpeg = require('ffmpeg-static');

const URL_ = process.argv[2] || 'http://localhost:8788';
const FPS = 12;
const W = 1280, H = 720;
const VO = 'demo/vo';
const FRAMES = 'demo/frames';
const OUT = 'demo/out';

// Segment durations come from the rendered MP3s (CBR 128 kbps = 16000 B/s).
const dur = f => fs.statSync(path.join(VO, f)).size / 16000;

const SHOTS = [
  { id: '01_hook',       mp3: '01_hook.mp3' },
  { id: '02_meter',      mp3: '02_meter.mp3' },
  { id: '03_unlabelled', mp3: '03_unlabelled.mp3' },
  { id: '04_validation', mp3: '04_validation.mp3' },
  { id: '05_audit',      mp3: '05_audit.mp3' },
  { id: '06_close',      mp3: '06_close.mp3' },
];

const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

(async () => {
  for (const d of [FRAMES, OUT]) fs.rmSync(d, { recursive: true, force: true });
  for (const d of [FRAMES, OUT]) fs.mkdirSync(d, { recursive: true });

  const browser = await puppeteer.launch({
    args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required',
           `--window-size=${W},${H}`],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  await page.goto(URL_, { waitUntil: 'networkidle2', timeout: 60000 });
  await page.waitForFunction("document.querySelectorAll('#kpis .card').length >= 5",
                             { timeout: 20000 });

  // cleaner capture: hide native video chrome, we drive the timeline ourselves
  await page.addStyleTag({ content: `
    video::-webkit-media-controls { display:none !important; }
    #dl { box-shadow: 0 0 0 0 rgba(59,130,246,.7); }
  `});

  const tab = async k =>
    page.evaluate(t => document.querySelector(`.tabbtn[data-k="${t}"]`).click(), k);
  const scrollTo = async y => page.evaluate(v => window.scrollTo(0, v), y);
  const maxScroll = () =>
    page.evaluate(() => Math.max(0, document.body.scrollHeight - window.innerHeight));

  // Per-shot setup + per-frame update. p is progress 0..1.
  const setup = {
    '01_hook': async () => { await tab('video'); await scrollTo(0); },
    // 472 centres the video + speed strip block in a 720px viewport
    '02_meter': async () => {
      await tab('video');
      await page.evaluate(() => document.querySelectorAll('.clipbtn')[1].click());
      await new Promise(r => setTimeout(r, 700));
      await scrollTo(472);
    },
    '03_unlabelled': async () => {
      await tab('video');
      await page.evaluate(() => document.querySelectorAll('.clipbtn')[6].click());
      await new Promise(r => setTimeout(r, 700));
      await scrollTo(472);
    },
    '04_validation': async () => { await tab('validation'); await scrollTo(0); },
    '05_audit':      async () => { await tab('corpus');     await scrollTo(0); },
    '06_close':      async () => { await tab('episodes');   await scrollTo(0); },
  };

  const frame = {
    // gentle drift down the KPI row into the tab bar
    '01_hook': async p => scrollTo(Math.round(ease(p) * 150)),
    // sweep the playhead through the 60.1s outlier span
    '02_meter': async p => page.evaluate(t => {
      const v = document.getElementById('vid');
      if (v) v.currentTime = t;
    }, 5 + p * 69),
    // sweep the whole unlabelled episode
    '03_unlabelled': async p => page.evaluate(t => {
      const v = document.getElementById('vid');
      if (v) v.currentTime = t;
    }, p * 97),
    '04_validation': async (p, m) => scrollTo(Math.round(ease(p) * m)),
    '05_audit':      async (p, m) => scrollTo(Math.round(ease(p) * m)),
    // pan the table while the counts are read, then come back to the export
    // button and pulse it under the "exports as a filter" line
    '06_close': async (p, m) => {
      const btn = await page.evaluate(() => {
        const b = document.getElementById('dl');
        return b ? b.getBoundingClientRect().top + window.scrollY : 0;
      });
      const rest = Math.max(0, Math.round(btn - 200));
      const deep = Math.min(m, 420);
      if (p < 0.35) await scrollTo(Math.round(ease(p / 0.35) * deep));
      else if (p < 0.5) await scrollTo(Math.round(deep + ease((p - 0.35) / 0.15) * (rest - deep)));
      else await scrollTo(rest);
      const on = p >= 0.5 && Math.floor((p - 0.5) * 16) % 2 === 0;
      await page.evaluate(o => {
        const b = document.getElementById('dl');
        if (b) b.style.boxShadow = o ? '0 0 0 6px rgba(59,130,246,.45)' : 'none';
      }, on);
    },
  };

  let grand = 0;
  for (const shot of SHOTS) {
    const seconds = dur(shot.mp3);
    const n = Math.round(seconds * FPS);
    const dir = path.join(FRAMES, shot.id);
    fs.mkdirSync(dir, { recursive: true });

    await setup[shot.id]();
    await new Promise(r => setTimeout(r, 500));
    const m = await maxScroll();

    const t0 = Date.now();
    for (let i = 0; i < n; i++) {
      await frame[shot.id](n === 1 ? 0 : i / (n - 1), m);
      await page.screenshot({
        path: path.join(dir, String(i).padStart(5, '0') + '.jpg'),
        type: 'jpeg', quality: 82,
      });
    }
    grand += n;
    console.log(`${shot.id.padEnd(15)} ${seconds.toFixed(1)}s  ${n} frames  ` +
                `captured in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await browser.close();
  console.log(`\n${grand} frames captured. Encoding...\n`);

  // encode each shot, then concat video, concat audio, mux
  const run = a => execFileSync(ffmpeg, a, { stdio: ['ignore', 'ignore', 'pipe'] });
  for (const shot of SHOTS) {
    run(['-y', '-framerate', String(FPS),
         '-i', path.join(FRAMES, shot.id, '%05d.jpg'),
         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
         '-pix_fmt', 'yuv420p', '-r', '30',
         path.join(OUT, shot.id + '.mp4')]);
    console.log('encoded', shot.id);
  }

  const vlist = SHOTS.map(s => `file '${s.id}.mp4'`).join('\n');
  fs.writeFileSync(path.join(OUT, 'v.txt'), vlist);
  const alist = SHOTS.map(s => `file '${path.resolve(VO, s.mp3).replace(/\\/g, '/')}'`)
                     .join('\n');
  fs.writeFileSync(path.join(OUT, 'a.txt'), alist);

  run(['-y', '-f', 'concat', '-safe', '0', '-i', path.join(OUT, 'v.txt'),
       '-c', 'copy', path.join(OUT, 'video.mp4')]);
  run(['-y', '-f', 'concat', '-safe', '0', '-i', path.join(OUT, 'a.txt'),
       '-c', 'copy', path.join(OUT, 'audio.mp3')]);
  run(['-y', '-i', path.join(OUT, 'video.mp4'), '-i', path.join(OUT, 'audio.mp3'),
       '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
       'demo/FailureRadar_demo.mp4']);

  const size = fs.statSync('demo/FailureRadar_demo.mp4').size;
  console.log(`\ndemo/FailureRadar_demo.mp4  ${(size / 1e6).toFixed(1)} MB`);
})();
