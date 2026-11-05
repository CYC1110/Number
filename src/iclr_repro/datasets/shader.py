"""Preserved renderer; requires a user-provided local shader manifest."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
from .grating import sample_seed


from .. import paths
ROOT = paths.inside('data', 'assets', 'shaders')

def sample_seed(dataset_seed: int, sample_id: int) -> int:
    return int(np.random.SeedSequence([dataset_seed, sample_id]).generate_state(1)[0])

class ShaderRenderer:
    """Headless ModernGL implementation of the upstream shader render procedure.

    Takes a fixed manifest of real upstream programs. No synthetic stand-in is
    silently substituted if shader source data are unavailable.
    """
    def __init__(self, manifest: str, resolution: int = 256, dataset_seed: int = 20260916,
                 preload: bool = True):
        import moderngl
        self.gl = moderngl
        self.resolution, self.dataset_seed = resolution, dataset_seed
        path = Path(manifest).resolve()
        self.manifest = json.loads(path.read_text())
        self.records = self.manifest['programs']
        if not self.records:
            raise ValueError('No shader programs in manifest')
        self.base = path.parent
        self.ctx = moderngl.create_context(standalone=True, backend='egl', require=440, device_index=int(os.environ.get('SIMCLR_EGL_DEVICE_INDEX','0')))
        self.info = dict(self.ctx.info)
        self.fbo = self.ctx.simple_framebuffer((resolution, resolution), components=4)
        self.fbo.use()
        self.vbo = self.ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1], dtype='f4'))
        self.templates = json.loads((ROOT/'assets/shader_templates.json').read_text())
        self.programs = []
        self.audit_references = []
        self.render_count = 0
        if preload:
            for index in range(len(self.records)):
                self.programs.append(self.compile_record(index))
                if (index+1) % 512 == 0:
                    print(f'SHADER_COMPILE {index+1}/{len(self.records)}',flush=True)
            if any(s in self.info.get('GL_RENDERER','').lower() for s in ('llvmpipe','softpipe','software')):
                raise RuntimeError('Shaders requires hardware EGL; software renderer detected')
            # Capture and replay a distributed reference set, then guard readback periodically.
            for index in np.linspace(0,len(self.records)-1,16,dtype=int):
                t=float(np.random.default_rng(sample_seed(20260916,int(index))).uniform(.1,10))
                image=self.render_program(self.programs[index],int(index),t)
                self.audit_references.append({'id':self.records[index]['id'],'index':int(index),
                    'program_slot':int(index),'sha256':hashlib.sha256(image.tobytes()).hexdigest()})
            self.check_reference_images()
            self.fingerprint={'gl_vendor':self.info.get('GL_VENDOR'),'gl_renderer':self.info.get('GL_RENDERER'),
                'gl_version':self.info.get('GL_VERSION'),'references':self.audit_references}
            expected=os.environ.get('SIMCLR_SHADER_FINGERPRINT')
            if expected and json.loads(expected)!=self.fingerprint:
                raise RuntimeError('Shader environment/reference images changed; use the original GPU stack for resume')

    def compile_record(self, index):
        rec = self.records[index]
        path=self.base/rec['assembled_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['assembled_sha256']:
            raise RuntimeError('Shader source changed: '+rec['id'])
        fragment=path.read_text(encoding='utf-8')
        prog = self.ctx.program(vertex_shader=self.templates['vertex_shader_moderngl'],
                                fragment_shader=fragment)
        vao = self.ctx.simple_vertex_array(prog, self.vbo, 'in_vert')
        return prog, vao, rec['kind']

    def __call__(self, sample_id: int):
        if self.audit_references and self.render_count and self.render_count % 4096 == 0:
            self.check_reference_images()
        self.render_count += 1
        r = np.random.default_rng(sample_seed(self.dataset_seed, sample_id))
        index = int(r.integers(len(self.programs)))
        t = float(r.uniform(.1, 10))  # official live-rendering interval
        return self.render_program(self.programs[index],sample_id,t)

    def check_reference_images(self):
        for row in self.audit_references:
            t=float(np.random.default_rng(sample_seed(20260916,row['index'])).uniform(.1,10))
            a=self.render_program(self.programs[row['program_slot']],row['index'],t)
            if hashlib.sha256(a.tobytes()).hexdigest()!=row['sha256']:
                raise RuntimeError(f"Shader reference image changed: {row['id']}")
        print(f'SHADER_REFERENCE_CHECK_PASS {len(self.audit_references)} after {self.render_count} images',flush=True)

    def render_program(self, program, sample_id, t):
        prog, vao, kind = program
        n = self.resolution
        if kind == 'shadertoy':
            uniforms = {'iResolution': (float(n),)*3, 'iTime': t, 'iMouse': (0.,)*4,
                        'iFrame': int(sample_id), 'iDate': (3.,10.,3.,2021.),
                        'iSampleRate': .1, 'iTimeDelta': .1, 'iFrameRate': 10.}
        else:
            # Set both classic and compact spellings, if active in this program.
            uniforms = {'resolution': (float(n),)*2, 'r': (float(n),)*2,
                        'time': t, 't': t, 'mouse': (0.,0.), 'm': (0.,0.),
                        'frame': int(sample_id), 'f': int(sample_id),
                        'backbuffer': 0, 'b': 0}
        for name, value in uniforms.items():
            if name in prog:
                prog[name].value = value
        vao.render(mode=self.gl.TRIANGLES)
        # On the tested NVIDIA 595.84 EGL stack, readback without an explicit
        # finish intermittently returned stale/garbled pixels across programs.
        # Keep the synchronization inside the timed generation path.
        self.ctx.finish()
        a = np.frombuffer(self.fbo.read(components=3, alignment=1), dtype=np.uint8).reshape(n,n,3)
        return np.ascontiguousarray(a[::-1])

    def close(self):
        for p, v, _ in self.programs:
            v.release(); p.release()
        self.vbo.release(); self.fbo.release(); self.ctx.release()
