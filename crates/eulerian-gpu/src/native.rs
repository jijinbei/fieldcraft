//! The device side of [`GpuSolid`] (native targets).

use std::time::{Duration, Instant};

use fieldcraft_eulerian::{CPU_RINGS, Error, NodeKind, Solid};
use fieldcraft_gpu::select::{self, Policy};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("interior.wgsl");
const WG: u32 = 256;
const MAX_GROUPS_X: u32 = 32768;
/// How long one wait for the device may take before the step gives up on it.
const DEVICE_TIMEOUT: Duration = Duration::from_secs(120);
/// Rings beyond [`CPU_RINGS`] that the stepped nodes keep from the surface when the work is
/// split, so the surface may move for a while before it has to be split again.
const MARGIN: usize = 2;
/// Steps between checks that the stepped nodes are still deep inside.
const CHECK_EVERY: usize = 10;
/// Steps per submission when the device steps every node (no exchange).
const CHUNK: usize = 64;

fn gpu_error(e: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("GPU: {e}"))
}

fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_of(n: usize, what: &str) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| Error::TooLarge(format!("{n} {what} are more than the GPU indexes")))
}

/// Workgroup counts (x, y) covering `count` items.
fn groups(count: usize) -> Result<(u32, u32), Error> {
    let total = u32_of(count.div_ceil(WG as usize).max(1), "workgroups")?;
    let gx = total.min(MAX_GROUPS_X);
    let gy = total.div_ceil(gx);
    if gy > 65535 {
        return Err(Error::TooLarge("too many nodes for one dispatch".into()));
    }
    Ok((gx, gy))
}

/// How a [`GpuSolid`] runs.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Form the stress from small quantities (the default); `false` evaluates the CPU's
    /// formulas as written, for measuring what single precision loses to cancellation.
    pub stable: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { stable: true }
    }
}

/// Where the time of [`GpuSolid::run`] went (wall clock, seconds).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    pub steps: usize,
    /// Packing the CPU's seam values and recording the device's step.
    pub submit: f64,
    /// The CPU's step of its own nodes (the device runs meanwhile).
    pub cpu: f64,
    /// Waiting for the device after the CPU's step, and reading its seam values back.
    pub wait: f64,
    /// Checking that the device's nodes are still deep, and splitting the work again.
    pub checks: f64,
}

/// A solid with its deep interior stepped on the GPU.
pub struct GpuSolid {
    solid: Solid,
    stable: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    state: [wgpu::Buffer; 2],
    flux: wgpu::Buffer,
    status: wgpu::Buffer,
    split: Option<Split>,
    /// Which state buffer holds the current step.
    current: usize,
    n: usize,
    cells: (u32, u32),
    since_check: usize,
    timings: Timings,
}

/// The nodes the device steps and the buffers that depend on them.
struct Split {
    remote: Vec<bool>,
    stepped: usize,
    upload: Vec<u32>,
    download: Vec<u32>,
    upload_values: wgpu::Buffer,
    download_values: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: [wgpu::BindGroup; 2],
    upload_groups: (u32, u32),
    download_groups: (u32, u32),
    _flags: wgpu::Buffer,
    _body: wgpu::Buffer,
    _upload_nodes: wgpu::Buffer,
    _download_nodes: wgpu::Buffer,
}

impl GpuSolid {
    /// `solid` on the device `FIELDCRAFT_GPU` picks.
    pub fn from_env(solid: Solid, options: Options) -> Result<GpuSolid, Error> {
        let policy = Policy::from_env().map_err(gpu_error)?;
        GpuSolid::new(solid, policy, options)
    }

    /// `solid` (mapped boundary) on the device `policy` picks; the work is split at once.
    pub fn new(solid: Solid, policy: Policy, options: Options) -> Result<GpuSolid, Error> {
        let Some((adapter, tier)) = select::adapter(policy).map_err(gpu_error)? else {
            return Err(Error::Invalid(format!("{}=cpu: no GPU device; step on the CPU", select::ENV)));
        };
        let info = adapter.get_info();
        let have = adapter.limits();
        let mut limits = wgpu::Limits::default().using_resolution(have.clone());
        limits.max_buffer_size = have.max_buffer_size;
        limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage = have.max_storage_buffers_per_shader_stage.max(limits.max_storage_buffers_per_shader_stage);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fieldcraft eulerian"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(gpu_error)?;
        let adapter_name = format!("{}{}", info.name, if tier == select::Tier::Software { " (software)" } else { "" });
        let n = solid.grid().len();
        let cap = device.limits().max_storage_buffer_binding_size.min(device.limits().max_buffer_size);
        if 16 * n as u64 > cap {
            return Err(Error::TooLarge(format!("{n} nodes need {} MB per buffer; this GPU allows {} MB", (16 * n) >> 20, cap >> 20)));
        }
        let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Validation].map(|f| device.push_error_scope(f));
        let empty = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let state = [empty("eulerian state a", 16 * n as u64), empty("eulerian state b", 16 * n as u64)];
        let flux = empty("eulerian flux", 16 * n as u64);
        let status = empty("eulerian status", 16);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eulerian params"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fieldcraft eulerian"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fieldcraft eulerian"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, read),
                entry(2, write),
                entry(3, write),
                entry(4, write),
                entry(5, read),
                entry(6, read),
                entry(7, read),
                entry(8, read),
                entry(9, write),
                entry(10, write),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fieldcraft eulerian"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let stable = if options.stable { 1.0 } else { 0.0 };
        let pipelines = ["upload", "faces", "momentum", "advect", "download"]
            .map(|name| {
                let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: wgpu::PipelineCompilationOptions { constants: &[("STABLE", stable)], ..Default::default() },
                    cache: None,
                });
                (name, p)
            })
            .to_vec();
        for scope in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(gpu_error(e));
            }
        }
        let mut g = GpuSolid {
            solid,
            stable: options.stable,
            device,
            queue,
            adapter: adapter_name,
            pipelines,
            layout,
            params,
            state,
            flux,
            status,
            split: None,
            current: 0,
            n,
            cells: groups(n)?,
            since_check: 0,
            timings: Timings::default(),
        };
        g.resplit()?;
        Ok(g)
    }

    /// The adapter's name.
    pub fn adapter_name(&self) -> &str {
        &self.adapter
    }

    /// Whether the stress is formed from small quantities ([`Options::stable`]).
    pub fn stable(&self) -> bool {
        self.stable
    }

    /// The solid as the CPU holds it: current near the surface and on the seam; the device's
    /// nodes are as of the last [`GpuSolid::sync`] or split.
    pub fn solid(&self) -> &Solid {
        &self.solid
    }

    /// Nodes the device steps, and nodes exchanged per step (sent, received).
    pub fn counts(&self) -> (usize, usize, usize) {
        self.split.as_ref().map_or((0, 0, 0), |s| (s.stepped, s.upload.len(), s.download.len()))
    }

    /// Where the time went so far.
    pub fn timings(&self) -> Timings {
        self.timings
    }

    /// Reads every device node back into the solid, which is then current everywhere.
    pub fn sync(&mut self) -> Result<&Solid, Error> {
        let values = self.read_f32(&self.state[self.current], 4 * self.n)?;
        if let Some(split) = &self.split {
            for k in 0..self.n {
                if split.remote.get(k).copied().unwrap_or(false)
                    && let Some(v) = values.get(4 * k..4 * k + 4)
                {
                    self.solid.set_node_state(k, [f64::from(v[0]), f64::from(v[1])], [f64::from(v[2]), f64::from(v[3])])?;
                }
            }
        }
        Ok(&self.solid)
    }

    /// The solid, current everywhere, and stepped on the CPU from here on.
    pub fn into_solid(mut self) -> Result<Solid, Error> {
        self.sync()?;
        let mut solid = self.solid;
        solid.set_remote(None)?;
        Ok(solid)
    }

    /// Advances `steps` steps.
    pub fn run(&mut self, steps: usize) -> Result<(), Error> {
        let mut left = steps;
        while left > 0 {
            // With no node on the CPU there is nothing to exchange and no surface to watch: the
            // device takes the steps in chunks.
            let all = self.split.as_ref().is_some_and(|s| s.upload.is_empty() && s.download.is_empty() && s.stepped > 0);
            if !all && self.since_check >= CHECK_EVERY {
                let t = Instant::now();
                self.check()?;
                self.timings.checks += t.elapsed().as_secs_f64();
            }
            let chunk = if all { left.min(CHUNK) } else { 1 };
            self.step(chunk, !all)?;
            left -= chunk;
            self.since_check += chunk;
        }
        Ok(())
    }

    /// `count` steps; with `exchange`, one step with the CPU stepping its nodes alongside.
    fn step(&mut self, count: usize, exchange: bool) -> Result<(), Error> {
        let t0 = Instant::now();
        let Some(split) = &self.split else {
            return Err(gpu_error("the work is not split"));
        };
        // The CPU's values on the seam.
        if exchange && !split.upload.is_empty() {
            let mut values = Vec::with_capacity(4 * split.upload.len());
            for &k in &split.upload {
                let (z, v) = self.solid.node_state(k as usize).unwrap_or(([0.0; 2], [0.0; 2]));
                values.extend([z[0] as f32, z[1] as f32, v[0] as f32, v[1] as f32]);
            }
            self.queue.write_buffer(&split.upload_values, 0, &bytes_f32(&values));
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eulerian steps") });
        let mut kick = self.solid.next_kick();
        let dt = self.solid.dt();
        for _ in 0..count {
            // The uniform (its kick is Δt/2 in the very first step) is written before the
            // submission that uses it.
            self.queue.write_buffer(&self.params, 0, &self.params_bytes(kick, split)?);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("eulerian step"), timestamp_writes: None });
                let bind = split.bind.get(self.current).ok_or_else(|| gpu_error("no bind group"))?;
                pass.set_bind_group(0, bind, &[]);
                if exchange && !split.upload.is_empty() {
                    self.dispatch(&mut pass, "upload", split.upload_groups)?;
                }
                for name in ["faces", "momentum", "advect"] {
                    self.dispatch(&mut pass, name, self.cells)?;
                }
                if exchange && !split.download.is_empty() {
                    self.dispatch(&mut pass, "download", split.download_groups)?;
                }
            }
            if count > 1 {
                // Each step is its own submission, after its uniform.
                let done = std::mem::replace(
                    &mut enc,
                    self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eulerian steps") }),
                );
                self.queue.submit([done.finish()]);
            }
            self.current ^= 1;
            kick = dt;
        }
        let bytes = 16 * split.download.len() as u64;
        if exchange && bytes > 0 {
            enc.copy_buffer_to_buffer(&split.download_values, 0, &split.staging, 0, bytes);
        }
        enc.copy_buffer_to_buffer(&self.status, 0, &split.staging, bytes, 16);
        self.queue.submit([enc.finish()]);
        self.timings.submit += t0.elapsed().as_secs_f64();

        // The CPU's own nodes, while the device runs.
        let t1 = Instant::now();
        if exchange {
            self.solid.run(1)?;
        } else {
            self.solid.run(count)?;
        }
        self.timings.cpu += t1.elapsed().as_secs_f64();

        // The device's new values on the seam, and its status.
        let t2 = Instant::now();
        let read = self.map(&split.staging, bytes + 16)?;
        let word = |at: usize| read.get(at..at + 4).and_then(|b| b.try_into().ok()).map_or(0, u32::from_le_bytes);
        if word(bytes as usize) != 0 {
            return Err(Error::Diverged(format!(
                "the reference map folded or a value stopped being a number on the GPU by t = {}",
                self.solid.time()
            )));
        }
        if exchange {
            for (i, &k) in split.download.iter().enumerate() {
                let v = |c: usize| f64::from(f32::from_bits(word(16 * i + 4 * c)));
                self.solid.set_node_state(k as usize, [v(0), v(1)], [v(2), v(3)])?;
            }
        }
        self.timings.wait += t2.elapsed().as_secs_f64();
        self.timings.steps += count;
        Ok(())
    }

    fn params_bytes(&self, kick: f64, split: &Split) -> Result<Vec<u8>, Error> {
        let c = self.solid.step_constants();
        let periodic = u32::from(c.periodic[0]) | (u32::from(c.periodic[1]) << 1);
        let mut b = bytes_u32(&[u32_of(c.nx, "nodes")?, u32_of(c.ny, "nodes")?, periodic, u32::from(c.neo_hookean)]);
        b.extend(bytes_f32(&[
            c.h as f32,
            c.dt as f32,
            kick as f32,
            c.damping as f32,
            c.lambda as f32,
            c.mu as f32,
            c.density as f32,
            c.viscosity as f32,
        ]));
        b.extend(bytes_u32(&[u32_of(split.upload.len(), "nodes")?, u32_of(split.download.len(), "nodes")?, self.cells.0, 0]));
        Ok(b)
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, name: &str, (gx, gy): (u32, u32)) -> Result<(), Error> {
        let p = self.pipelines.iter().find(|(n, _)| *n == name).ok_or_else(|| gpu_error(format!("no kernel {name}")))?;
        pass.set_pipeline(&p.1);
        pass.dispatch_workgroups(gx, gy, 1);
        Ok(())
    }

    /// Checks that the device's nodes are still deep inside; if not, splits the work again.
    fn check(&mut self) -> Result<(), Error> {
        self.since_check = 0;
        let deep = self.solid.deep_interior(CPU_RINGS);
        let ok = self.split.as_ref().is_some_and(|s| s.remote.iter().zip(&deep).all(|(&r, &d)| !r || d));
        if ok { Ok(()) } else { self.resplit() }
    }

    /// Reads the device's nodes back, chooses them again ([`CPU_RINGS`] + [`MARGIN`] deep) and
    /// sends the whole state.
    fn resplit(&mut self) -> Result<(), Error> {
        if self.split.is_some() {
            self.sync()?;
        }
        self.solid.set_remote(None)?;
        let n = self.n;
        let grid = self.solid.grid().clone();
        let remote = self.solid.deep_interior(CPU_RINGS + MARGIN);
        let known: Vec<bool> = (0..n).map(|k| self.solid.kind(k).is_some_and(|k| k != NodeKind::Empty)).collect();
        // Nodes within two rings (the stencil's reach) of a node of the other kind.
        let near =
            |k: usize, other: &dyn Fn(usize) -> bool| -> bool { (-2..=2).any(|dj| (-2..=2).any(|di| grid.neighbour(k, di, dj).is_some_and(other))) };
        let upload: Vec<u32> = (0..n).filter(|&k| known[k] && !remote[k] && near(k, &|m| remote[m])).map(|k| k as u32).collect();
        let download: Vec<u32> = (0..n).filter(|&k| remote[k] && near(k, &|m| known[m] && !remote[m])).map(|k| k as u32).collect();
        // Faces the device computes: those with a stepped node at either end.
        let mut flags = vec![0u32; n];
        for k in 0..n {
            if remote[k] {
                flags[k] |= 1;
            }
            for d in 0..2 {
                let b = if d == 0 { grid.neighbour(k, 1, 0) } else { grid.neighbour(k, 0, 1) };
                if let Some(b) = b
                    && (remote[k] || remote[b])
                {
                    flags[k] |= 2 << d;
                }
            }
        }
        let mut body = vec![0.0f32; 2 * n];
        let mut state = vec![0.0f32; 4 * n];
        for k in 0..n {
            if remote[k]
                && let Some(f) = self.solid.body_force_at(grid.position(k))
            {
                body[2 * k] = f[0] as f32;
                body[2 * k + 1] = f[1] as f32;
            }
            let (z, v) = self.solid.node_state(k).unwrap_or(([0.0; 2], [0.0; 2]));
            state[4 * k..4 * k + 4].copy_from_slice(&[z[0] as f32, z[1] as f32, v[0] as f32, v[1] as f32]);
        }
        let stepped = remote.iter().filter(|&&r| r).count();
        self.solid.set_remote(Some(remote.clone()))?;

        let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Validation].map(|f| self.device.push_error_scope(f));
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let init = |label: &str, bytes: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16][..] } else { bytes },
                usage,
            })
        };
        let flags_buf = init("eulerian flags", &bytes_u32(&flags));
        let body_buf = init("eulerian body force", &bytes_f32(&body));
        let upload_nodes = init("eulerian upload nodes", &bytes_u32(&upload));
        let download_nodes = init("eulerian download nodes", &bytes_u32(&download));
        let upload_values = init("eulerian upload values", &vec![0u8; 16 * upload.len()]);
        let download_values = init("eulerian download values", &vec![0u8; 16 * download.len()]);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eulerian readback"),
            size: 16 * download.len() as u64 + 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.current = 0;
        self.queue.write_buffer(&self.state[0], 0, &bytes_f32(&state));
        self.queue.write_buffer(&self.state[1], 0, &bytes_f32(&state));
        self.queue.write_buffer(&self.status, 0, &bytes_u32(&[0, 0, 0, 0]));
        let bind = [0usize, 1].map(|c| {
            fn e(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
            }
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fieldcraft eulerian"),
                layout: &self.layout,
                entries: &[
                    e(0, &self.params),
                    e(1, &flags_buf),
                    e(2, &self.state[c]),
                    e(3, &self.state[1 - c]),
                    e(4, &self.flux),
                    e(5, &body_buf),
                    e(6, &upload_nodes),
                    e(7, &upload_values),
                    e(8, &download_nodes),
                    e(9, &download_values),
                    e(10, &self.status),
                ],
            })
        });
        for scope in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(gpu_error(e));
            }
        }
        self.split = Some(Split {
            remote,
            stepped,
            upload_groups: groups(upload.len())?,
            download_groups: groups(download.len())?,
            upload,
            download,
            upload_values,
            download_values,
            staging,
            bind,
            _flags: flags_buf,
            _body: body_buf,
            _upload_nodes: upload_nodes,
            _download_nodes: download_nodes,
        });
        self.since_check = 0;
        Ok(())
    }

    /// Maps `size` bytes of the staging buffer `buf` (already copied into) and reads them.
    fn map(&self, buf: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, Error> {
        let slice = buf.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if let Err(e) = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(DEVICE_TIMEOUT) }) {
            return Err(gpu_error(format!("the device did not answer within {} s: {e}", DEVICE_TIMEOUT.as_secs())));
        }
        rx.try_recv().map_err(|_| gpu_error("the readback did not complete"))?.map_err(gpu_error)?;
        let bytes = slice.get_mapped_range().map_err(gpu_error)?.to_vec();
        buf.unmap();
        Ok(bytes)
    }

    /// Copies `count` floats of `src` back to the host.
    fn read_f32(&self, src: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, Error> {
        let size = (4 * count as u64).max(16);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eulerian readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eulerian readback") });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let bytes = self.map(&staging, size)?;
        Ok(bytes.as_chunks::<4>().0.iter().take(count).map(|b| f32::from_le_bytes(*b)).collect())
    }
}
