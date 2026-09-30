// Corporis
// Apache License, Version 2.0
//
// MediaPipe's pose and hand landmarkers in the page, for the pose from a picture. The page loads MediaPipe's code,
// its WebAssembly and its models from their CDNs the first time a picture comes in, so everything else works offline.

import { fromMediapipe, type Landmarks } from './pose_from_image.ts';

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const MODELS = {
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
  hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
};

interface Point { x: number; y: number; z: number; visibility?: number }
interface Found { landmarks: Point[][]; worldLandmarks: Point[][] }
interface Landmarker { detect(image: TexImageSource): Found }

let loading: Promise<{ pose: Landmarker; hand: Landmarker }> | null = null;
function landmarkers() {
  if (!loading) {
    loading = (async () => {
      const url = `${CDN}/vision_bundle.mjs`;
      const vision = await import(/* @vite-ignore */ url);
      const files = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
      const make = (cls: any, model: string, extra: object) =>
        cls.createFromOptions(files, { baseOptions: { modelAssetPath: model, delegate: 'CPU' }, runningMode: 'IMAGE', ...extra });
      const [pose, hand] = await Promise.all([make(vision.PoseLandmarker, MODELS.pose, { numPoses: 1 }), make(vision.HandLandmarker, MODELS.hand, { numHands: 2 })]);
      return { pose, hand };
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

// the landmarks of the most prominent figure in a picture (anny's frame), and which hands MediaPipe found
export async function detect(image: TexImageSource & { width: number; height: number }): Promise<{ landmarks: Landmarks; hands: string[] }> {
  const { pose, hand } = await landmarkers();
  const found = pose.detect(image);
  if (!found.worldLandmarks.length) throw new Error('No figure found in the picture.');
  const img = found.landmarks[0];
  const landmarks: Landmarks = {
    body: fromMediapipe(found.worldLandmarks[0]), hands: {},
    image: img.map((q) => [q.x * image.width, q.y * image.height]), visibility: img.map((q) => q.visibility ?? 1),
  };
  // MediaPipe names a hand's side as seen in a mirror; the nearest wrist in the picture is surer, and a hand belongs
  // to a wrist only within most of a forearm's length of it
  const wrists: Record<string, Point> = { '.L': img[15], '.R': img[16] }, elbows: Record<string, Point> = { '.L': img[13], '.R': img[14] };
  const px = (a: Point, b: Point) => Math.hypot((a.x - b.x) * image.width, (a.y - b.y) * image.height);
  const hands = hand.detect(image), taken = new Set<string>();
  hands.landmarks.forEach((points, i) => {
    const at = points[0];
    let side = '', best = Infinity;
    for (const s of ['.L', '.R']) {
      if (taken.has(s)) continue;
      const d = Math.hypot((wrists[s].x - at.x) * image.width, (wrists[s].y - at.y) * image.height);
      if (d < best) { best = d; side = s; }
    }
    if (!side || best > 0.6 * px(wrists[side], elbows[side])) return;
    taken.add(side);
    landmarks.hands[side] = fromMediapipe(hands.worldLandmarks[i]);
  });
  return { landmarks, hands: [...taken] };
}
