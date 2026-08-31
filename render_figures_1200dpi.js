const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const dir = __dirname;

(async () => {
  const files = fs.readdirSync(dir).filter((name) => name.endsWith('.svg'));
  for (const name of files) {
    const source = path.join(dir, name);
    const target = path.join(dir, name.replace(/\.svg$/i, '_1200dpi.png'));
    await sharp(source, { density: 1200, limitInputPixels: false })
      .resize({ width: 8400, withoutEnlargement: false })
      .png({ compressionLevel: 9, adaptiveFiltering: true })
      .withMetadata({ density: 1200 })
      .toFile(target);
    const meta = await sharp(target, { limitInputPixels: false }).metadata();
    console.log(`${path.basename(target)}: ${meta.width} x ${meta.height}, density=${meta.density}`);
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
