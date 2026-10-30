import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { Stage } from '../../../lib/components/Stage';
import { Captions } from '../../../lib/components/Captions';
import { cueFrame, parseCueDocument } from '../../../lib/cues';
import cueData from './cues.json';

const cues = parseCueDocument(cueData);
const settings = { x: 250, y: 180 };
const captions = { x: 540, y: 330 };
const apply = { x: 1080, y: 600 };

export const Demo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const openAt = cueFrame({ document: cues, name: 'openSettings', fps });
  const chooseAt = cueFrame({ document: cues, name: 'chooseCaptions', fps });
  const applyAt = cueFrame({ document: cues, name: 'applyChanges', fps });

  return (
    <AbsoluteFill>
      <Stage
        url="demo.example/settings"
        camera={[
          { f: 0, z: 1, x: 720, y: 405 },
          { f: chooseAt - 20, z: 1, x: 720, y: 405 },
          { f: chooseAt, z: 1.3, x: 740, y: 420 },
        ]}
        path={[
          { f: 0, x: 1000, y: 200 },
          { f: Math.max(0, openAt - 18), x: 1000, y: 200 },
          { f: openAt, ...settings },
          { f: chooseAt - 18, ...settings },
          { f: chooseAt, ...captions },
          { f: applyAt - 18, ...captions },
          { f: applyAt, ...apply },
        ]}
        clicks={[openAt, chooseAt, applyAt]}
      >
        <div style={{ padding: 80, fontFamily: 'system-ui', color: '#243653', height: '100%', boxSizing: 'border-box', background: '#fff' }}>
          <h1 style={{ fontSize: 42 }}>Speech-linked product demo</h1>
          <div style={{ position: 'absolute', left: settings.x - 90, top: settings.y - 30, padding: '16px 28px', background: '#e8eef8', borderRadius: 10, fontSize: 24 }}>
            Settings
          </div>
          {frame >= openAt && (
            <div style={{ position: 'absolute', left: 420, top: 260, width: 700, height: 400, border: '2px solid #dfe5f0', borderRadius: 16, padding: 38, boxSizing: 'border-box' }}>
              <div style={{ fontSize: 30 }}>Captions {frame >= chooseAt ? 'enabled' : 'disabled'}</div>
              <div style={{ marginTop: 35, fontSize: 22, color: '#62708a' }}>The cursor and camera follow named words.</div>
              <div style={{ position: 'absolute', right: 0, bottom: 30, width: 80, textAlign: 'center', background: '#3671e0', color: '#fff', padding: '12px 0', borderRadius: 8 }}>
                {frame >= applyAt ? 'Saved' : 'Apply'}
              </div>
            </div>
          )}
        </div>
      </Stage>
      <Captions src="narration.captions.json" color="#243653" background="rgba(255,255,255,0.95)" bottom={50} />
    </AbsoluteFill>
  );
};
