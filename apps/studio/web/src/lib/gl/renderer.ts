/** A WebGL2 renderer for one scalar field and one camera frame underlay.
 *
 * The field is a float texture sampled with bilinear interpolation done in the
 * shader, so cells the quantity leaves undefined (NaN, sent as a sentinel)
 * stay transparent instead of bleeding into their neighbours. Colours come
 * from a 256-entry lookup texture and a [vmin, vmax] window, so changing the
 * colormap or the range costs one uniform, not a re-upload.
 *
 * Without a WebGL2 context (jsdom, an old browser) every method is a no-op
 * and `available` is false; the scene then falls back to a CPU raster.
 */

import type { View } from "../transform";

const MISSING = -1e30;

const VERTEX = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos;
  gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);
}`;

const FIELD_FRAGMENT = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_field;
uniform sampler2D u_lut;
uniform vec2 u_size;      // nx, ny
uniform vec2 u_range;     // vmin, vmax
uniform vec4 u_window;    // µm extent of the screen: x0, x1, y0, y1
uniform vec4 u_extent;    // µm extent of the field: x0, x1, y0, y1
uniform float u_opacity;
const float MISSING = -1e29;

float texel(ivec2 ij) {
  ij = clamp(ij, ivec2(0), ivec2(u_size) - 1);
  return texelFetch(u_field, ij, 0).r;
}

void main() {
  float xu = mix(u_window.x, u_window.y, v_uv.x);
  float yu = mix(u_window.z, u_window.w, v_uv.y);
  vec2 f = vec2((xu - u_extent.x) / (u_extent.y - u_extent.x),
                (yu - u_extent.z) / (u_extent.w - u_extent.z));
  if (any(lessThan(f, vec2(0.0))) || any(greaterThan(f, vec2(1.0)))) discard;
  vec2 p = f * (u_size - 1.0);
  ivec2 ij = ivec2(floor(p));
  vec2 a = fract(p);
  float v00 = texel(ij), v10 = texel(ij + ivec2(1, 0));
  float v01 = texel(ij + ivec2(0, 1)), v11 = texel(ij + ivec2(1, 1));
  float nearest = texel(ivec2(floor(p + 0.5)));
  float value;
  if (v00 < MISSING || v10 < MISSING || v01 < MISSING || v11 < MISSING) {
    if (nearest < MISSING) discard;
    value = nearest;
  } else {
    value = mix(mix(v00, v10, a.x), mix(v01, v11, a.x), a.y);
  }
  float t = clamp((value - u_range.x) / max(u_range.y - u_range.x, 1e-30), 0.0, 1.0);
  vec3 rgb = texture(u_lut, vec2(t, 0.5)).rgb;
  o_color = vec4(rgb * u_opacity, u_opacity);
}`;

const IMAGE_FRAGMENT = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_image;
uniform vec4 u_window;
uniform vec4 u_extent;   // µm extent of the image: x0, x1, y0 (top row), y1 (bottom row)
uniform float u_opacity;
uniform float u_flip;    // 1 when the image's columns run against +x
void main() {
  float xu = mix(u_window.x, u_window.y, v_uv.x);
  float yu = mix(u_window.z, u_window.w, v_uv.y);
  float fx = (xu - u_extent.x) / (u_extent.y - u_extent.x);
  float fy = (yu - u_extent.z) / (u_extent.w - u_extent.z);
  if (fx < 0.0 || fx > 1.0 || fy < 0.0 || fy > 1.0) discard;
  if (u_flip > 0.5) fx = 1.0 - fx;
  vec4 c = texture(u_image, vec2(fx, fy));
  o_color = vec4(c.rgb * u_opacity, u_opacity);
}`;

export interface FieldUpload {
  plane: Float32Array;
  nx: number;
  ny: number;
  extent: [number, number, number, number];
}

export interface ImageUpload {
  image: TexImageSource;
  /** µm extent: x0, x1, yTopRow, yBottomRow (y of the first and last rows). */
  extent: [number, number, number, number];
  flip: boolean;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("could not create a shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader failed to compile: ${log}`);
  }
  return shader;
}

function link(gl: WebGL2RenderingContext, fragment: string): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error("could not create a program");
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`program failed to link: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

export class FieldRenderer {
  readonly available: boolean;
  private gl: WebGL2RenderingContext | null = null;
  private fieldProgram: WebGLProgram | null = null;
  private imageProgram: WebGLProgram | null = null;
  private fieldTexture: WebGLTexture | null = null;
  private lutTexture: WebGLTexture | null = null;
  private imageTexture: WebGLTexture | null = null;
  private field: FieldUpload | null = null;
  private image: ImageUpload | null = null;
  private range: [number, number] = [0, 1];
  private fieldOpacity = 1;
  private imageOpacity = 1;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", {
      premultipliedAlpha: true,
      antialias: false,
      alpha: true,
    });
    this.available = gl !== null;
    if (!gl) return;
    this.gl = gl;
    this.fieldProgram = link(gl, FIELD_FRAGMENT);
    this.imageProgram = link(gl, IMAGE_FRAGMENT);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    for (const program of [this.fieldProgram, this.imageProgram]) {
      const location = gl.getAttribLocation(program, "a_pos");
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    }
    this.fieldTexture = gl.createTexture();
    this.lutTexture = gl.createTexture();
    this.imageTexture = gl.createTexture();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  setField(upload: FieldUpload | null): void {
    const gl = this.gl;
    this.field = upload;
    if (!gl || !upload) return;
    const data = new Float32Array(upload.plane.length);
    for (let i = 0; i < data.length; i++) {
      const v = upload.plane[i];
      data[i] = Number.isFinite(v) ? v : MISSING;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, upload.nx, upload.ny, 0, gl.RED, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  setLut(lut: Uint8Array): void {
    const gl = this.gl;
    if (!gl) return;
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, lut.length / 4, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  setRange(range: [number, number]): void {
    this.range = range;
  }

  setOpacity(field: number, image: number): void {
    this.fieldOpacity = field;
    this.imageOpacity = image;
  }

  setImage(upload: ImageUpload | null): void {
    const gl = this.gl;
    this.image = upload;
    if (!gl || !upload) return;
    gl.bindTexture(gl.TEXTURE_2D, this.imageTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, upload.image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Draw the image then the field for the given view; the canvas size is
   * taken as the device-pixel size. */
  draw(view: View, width: number, height: number, dpr: number): void {
    const gl = this.gl;
    if (!gl) return;
    const pw = Math.max(1, Math.round(width * dpr));
    const ph = Math.max(1, Math.round(height * dpr));
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw;
      this.canvas.height = ph;
    }
    gl.viewport(0, 0, pw, ph);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    // µm at the screen's left/right and bottom/top edges (uv y runs upward).
    const window: [number, number, number, number] = [
      (0 - view.tx) / view.scale,
      (width - view.tx) / view.scale,
      (view.ty - height) / view.scale,
      view.ty / view.scale,
    ];
    if (this.image && this.imageProgram && this.imageOpacity > 0) {
      const p = this.imageProgram;
      gl.useProgram(p);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.imageTexture);
      gl.uniform1i(gl.getUniformLocation(p, "u_image"), 0);
      gl.uniform4fv(gl.getUniformLocation(p, "u_window"), window);
      gl.uniform4fv(gl.getUniformLocation(p, "u_extent"), this.image.extent);
      gl.uniform1f(gl.getUniformLocation(p, "u_opacity"), this.imageOpacity);
      gl.uniform1f(gl.getUniformLocation(p, "u_flip"), this.image.flip ? 1 : 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    if (this.field && this.fieldProgram && this.fieldOpacity > 0) {
      const p = this.fieldProgram;
      gl.useProgram(p);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.fieldTexture);
      gl.uniform1i(gl.getUniformLocation(p, "u_field"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
      gl.uniform1i(gl.getUniformLocation(p, "u_lut"), 1);
      gl.uniform2f(gl.getUniformLocation(p, "u_size"), this.field.nx, this.field.ny);
      gl.uniform2fv(gl.getUniformLocation(p, "u_range"), this.range);
      gl.uniform4fv(gl.getUniformLocation(p, "u_window"), window);
      gl.uniform4fv(gl.getUniformLocation(p, "u_extent"), this.field.extent);
      gl.uniform1f(gl.getUniformLocation(p, "u_opacity"), this.fieldOpacity);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  dispose(): void {
    const gl = this.gl;
    if (!gl) return;
    gl.deleteTexture(this.fieldTexture);
    gl.deleteTexture(this.lutTexture);
    gl.deleteTexture(this.imageTexture);
    if (this.fieldProgram) gl.deleteProgram(this.fieldProgram);
    if (this.imageProgram) gl.deleteProgram(this.imageProgram);
    this.gl = null;
  }
}
