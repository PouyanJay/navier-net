import { chromium } from "playwright";
const OUT = process.argv[2];
const URL = "http://127.0.0.1:5173/projects/cfb01afad527412a874637ab6fc35d01/physics";
const browser = await chromium.launch();
for (const theme of ["dark", "light"]) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  await page.addInitScript((t) => {
    localStorage.setItem("naviernet-theme", t);
    localStorage.setItem("naviernet-aside-collapsed", "0");
  }, theme);
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1800);
  await page.screenshot({ path: `${OUT}/physics-${theme}.png` });
  await page.close();
}
await browser.close();
console.log("done");
