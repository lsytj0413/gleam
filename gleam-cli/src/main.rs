use candle_core::{Device, Tensor};
use candle_nn::ops::softmax_last_dim;

/// GPT-2 配置 (124M 参数)
pub struct GptConfig {
    pub vocab_size: usize,    // 词汇表大小
    pub context_length: usize, // 上下文长度
    pub emb_dim: usize,       // 嵌入维度
    pub n_heads: usize,       // 注意力头数量
    pub n_layers: usize,      // 层数
    pub drop_rate: f32,       // Dropout 率
    pub qkv_bias: bool,       // QKV 偏置
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}

/// Layer Normalization
pub struct DummyLayerNorm {
    scale: Tensor, // 缩放参数 γ: [emb_dim]
    shift: Tensor, // 偏移参数 β: [emb_dim]
    eps: f32,      // 防止除零的小常数
}

impl DummyLayerNorm {
    pub fn new(emb_dim: usize, device: &Device) -> anyhow::Result<Self> {
        // 初始化: γ = 1, β = 0
        let scale = Tensor::ones((emb_dim,), candle_core::DType::F32, device)?;
        let shift = Tensor::zeros((emb_dim,), candle_core::DType::F32, device)?;
        Ok(Self { scale, shift, eps: 1e-5 })
    }

    /// 前向传播
    /// x: [seq_len, emb_dim] -> [seq_len, emb_dim]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        // 计算均值: [seq_len, 1]
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        // 计算方差: [seq_len, 1]
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        // 归一化: x_norm = (x - mean) / sqrt(var + eps)
        let x_norm = x.broadcast_sub(&mean)?.broadcast_div(&var.add(&Tensor::new(&[self.eps], x.device())?.broadcast_as(var.shape())?)?.sqrt()?)?;
        // 缩放和平移: γ * x_norm + β
        Ok(x_norm.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

/// 前馈网络 (Feed Forward Network)
pub struct DummyFeedForward {
    w1: Tensor, // [emb_dim, 4 * emb_dim]
    w2: Tensor, // [4 * emb_dim, emb_dim]
}

impl DummyFeedForward {
    pub fn new(emb_dim: usize, device: &Device) -> anyhow::Result<Self> {
        let hidden_dim = 4 * emb_dim;
        // 简化初始化
        let w1_data: Vec<f32> = (0..emb_dim * hidden_dim).map(|i| i as f32 * 0.0001).collect();
        let w2_data: Vec<f32> = (0..hidden_dim * emb_dim).map(|i| i as f32 * 0.0001).collect();
        let w1 = Tensor::from_vec(w1_data, (emb_dim, hidden_dim), device)?;
        let w2 = Tensor::from_vec(w2_data, (hidden_dim, emb_dim), device)?;
        Ok(Self { w1, w2 })
    }

    /// 前向传播: GELU(x @ W1) @ W2
    /// x: [seq_len, emb_dim] -> [seq_len, emb_dim]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let hidden = x.matmul(&self.w1)?;
        let hidden = gelu(&hidden)?;
        Ok(hidden.matmul(&self.w2)?)
    }
}

/// GELU 激活函数
fn gelu(x: &Tensor) -> anyhow::Result<Tensor> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let x3 = x.sqr()?.mul(x)?;
    // x + 0.044715 * x^3
    let inner = x.add(&x3.affine(0.044715, 0.0)?)?;
    // sqrt(2/π) * inner
    let inner = inner.affine(sqrt_2_over_pi, 0.0)?;
    let tanh = inner.tanh()?;
    // 0.5 * x * (1 + tanh)
    Ok(x.mul(&tanh.affine(1.0, 1.0)?)?.affine(0.5, 0.0)?)
}

/// 多头因果注意力 (优化版)
pub struct DummyMultiHeadAttention {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    w_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    dropout_p: f32,
}

impl DummyMultiHeadAttention {
    pub fn new(d_in: usize, d_out: usize, num_heads: usize, dropout_p: f32, use_bias: bool, device: &Device) -> anyhow::Result<Self> {
        let head_dim = d_out / num_heads;
        let scale = 0.02_f32;

        let wq = random_weight(d_in, d_out, scale, device)?;
        let wk = random_weight(d_in, d_out, scale, device)?;
        let wv = random_weight(d_in, d_out, scale, device)?;
        let w_proj = random_weight(d_out, d_out, scale, device)?;

        // 注意: 简化起见，use_bias 暂不实现
        let _ = use_bias;

        Ok(Self { wq, wk, wv, w_proj, num_heads, head_dim, dropout_p })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (seq_len, _d_in) = (x.shape().dims()[0], x.shape().dims()[1]);
        let device = x.device();

        // 投影
        let q = x.matmul(&self.wq)?;
        let k = x.matmul(&self.wk)?;
        let v = x.matmul(&self.wv)?;

        // Reshape + Transpose: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        let q = q.reshape((seq_len, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let k = k.reshape((seq_len, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let v = v.reshape((seq_len, self.num_heads, self.head_dim))?.transpose(0, 1)?;

        // 注意力分数
        let attn_scores = q.matmul(&k.transpose(1, 2)?)?;
        let scale = (self.head_dim as f64).sqrt();
        let scaled_scores = attn_scores.affine(1.0 / scale, 0.0)?;

        // 因果掩码
        let mask = causal_mask(seq_len, device)?;
        let mask = mask.unsqueeze(0)?.broadcast_as((self.num_heads, seq_len, seq_len))?;
        let masked_scores = scaled_scores.add(&mask)?;

        // Softmax
        let attn_weights = softmax_last_dim(&masked_scores)?;

        // Dropout
        let attn_weights = if self.dropout_p > 0.0 {
            candle_nn::ops::dropout(&attn_weights, self.dropout_p)?
        } else {
            attn_weights
        };

        // 计算输出
        let context = attn_weights.matmul(&v)?;
        let context = context.transpose(0, 1)?.reshape((seq_len, self.num_heads * self.head_dim))?;
        Ok(context.matmul(&self.w_proj)?)
    }
}

/// Transformer Block
pub struct DummyTransformerBlock {
    attn: DummyMultiHeadAttention,
    ff: DummyFeedForward,
    norm1: DummyLayerNorm,
    norm2: DummyLayerNorm,
    drop_p: f32,
}

impl DummyTransformerBlock {
    pub fn new(cfg: &GptConfig, device: &Device) -> anyhow::Result<Self> {
        let attn = DummyMultiHeadAttention::new(
            cfg.emb_dim, cfg.emb_dim, cfg.n_heads, cfg.drop_rate, cfg.qkv_bias, device,
        )?;
        let ff = DummyFeedForward::new(cfg.emb_dim, device)?;
        let norm1 = DummyLayerNorm::new(cfg.emb_dim, device)?;
        let norm2 = DummyLayerNorm::new(cfg.emb_dim, device)?;

        Ok(Self { attn, ff, norm1, norm2, drop_p: cfg.drop_rate })
    }

    /// 前向传播 (Pre-LayerNorm)
    /// x = x + Attention(LayerNorm(x))
    /// x = x + FFN(LayerNorm(x))
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        // 注意力子层
        let norm_x = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&norm_x)?;
        let attn_out = if self.drop_p > 0.0 {
            candle_nn::ops::dropout(&attn_out, self.drop_p)?
        } else {
            attn_out
        };
        let x = x.add(&attn_out)?;

        // 前馈子层
        let norm_x = self.norm2.forward(&x)?;
        let ff_out = self.ff.forward(&norm_x)?;
        let ff_out = if self.drop_p > 0.0 {
            candle_nn::ops::dropout(&ff_out, self.drop_p)?
        } else {
            ff_out
        };
        Ok(x.add(&ff_out)?)
    }
}

/// Dummy GPT Model
pub struct DummyGptModel {
    tok_emb: Tensor,       // Token embedding: [vocab_size, emb_dim]
    pos_emb: Tensor,       // Position embedding: [context_length, emb_dim]
    blocks: Vec<DummyTransformerBlock>,
    final_norm: DummyLayerNorm,
    out_head: Tensor,      // Output head: [emb_dim, vocab_size]
    drop_p: f32,
}

impl DummyGptModel {
    pub fn new(cfg: &GptConfig, device: &Device) -> anyhow::Result<Self> {
        // Token embedding
        let tok_emb_data: Vec<f32> = (0..cfg.vocab_size * cfg.emb_dim)
            .map(|i| i as f32 * 0.0001)
            .collect();
        let tok_emb = Tensor::from_vec(tok_emb_data, (cfg.vocab_size, cfg.emb_dim), device)?;

        // Position embedding
        let pos_emb_data: Vec<f32> = (0..cfg.context_length * cfg.emb_dim)
            .map(|i| i as f32 * 0.0001)
            .collect();
        let pos_emb = Tensor::from_vec(pos_emb_data, (cfg.context_length, cfg.emb_dim), device)?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            blocks.push(DummyTransformerBlock::new(cfg, device)?);
        }

        // Final layer norm
        let final_norm = DummyLayerNorm::new(cfg.emb_dim, device)?;

        // Output head (与 token embedding 共享权重时，这里简化处理)
        let out_head_data: Vec<f32> = (0..cfg.emb_dim * cfg.vocab_size)
            .map(|i| i as f32 * 0.0001)
            .collect();
        let out_head = Tensor::from_vec(out_head_data, (cfg.emb_dim, cfg.vocab_size), device)?;

        Ok(Self { tok_emb, pos_emb, blocks, final_norm, out_head, drop_p: cfg.drop_rate })
    }

    /// 前向传播
    /// token_ids: [seq_len] -> logits: [seq_len, vocab_size]
    pub fn forward(&self, token_ids: &Tensor) -> anyhow::Result<Tensor> {
        let seq_len = token_ids.shape().dims()[0];
        let device = token_ids.device();

        // Token embedding: [seq_len, emb_dim]
        let tok_emb = self.tok_emb.index_select(token_ids, 0)?;

        // Position embedding: [seq_len, emb_dim]
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_tensor = Tensor::new(positions.as_slice(), device)?;
        let pos_emb = self.pos_emb.index_select(&pos_tensor, 0)?;

        // 组合 embedding
        let x = tok_emb.add(&pos_emb)?;
        let x = if self.drop_p > 0.0 {
            candle_nn::ops::dropout(&x, self.drop_p)?
        } else {
            x
        };

        // 通过所有 Transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // 最终 LayerNorm
        let x = self.final_norm.forward(&x)?;

        // 输出 logits: [seq_len, vocab_size]
        Ok(x.matmul(&self.out_head)?)
    }

    /// 获取参数量 (估算)
    pub fn param_count(&self) -> usize {
        let tok_emb = self.tok_emb.shape().elem_count();
        let pos_emb = self.pos_emb.shape().elem_count();
        let out_head = self.out_head.shape().elem_count();
        tok_emb + pos_emb + out_head + self.blocks.len() * 7 * self.out_head.shape().dims()[0]
    }
}

// ============ 辅助函数 ============

fn random_weight(rows: usize, cols: usize, scale: f32, device: &Device) -> anyhow::Result<Tensor> {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * scale).collect();
    Ok(Tensor::from_vec(data, (rows, cols), device)?)
}

fn causal_mask(seq_len: usize, device: &Device) -> anyhow::Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(mask_data, (seq_len, seq_len), device)?)
}

// ============ 主函数 ============

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let cfg = GptConfig::default();

    println!("=== DummyGPTModel Demo ===\n");
    println!("GPT-2 124M Configuration:");
    println!("  vocab_size:     {}", cfg.vocab_size);
    println!("  context_length: {}", cfg.context_length);
    println!("  emb_dim:        {}", cfg.emb_dim);
    println!("  n_heads:        {}", cfg.n_heads);
    println!("  n_layers:       {}", cfg.n_layers);
    println!("  drop_rate:      {}", cfg.drop_rate);
    println!("  qkv_bias:       {}", cfg.qkv_bias);

    // 创建模型
    println!("\nCreating model...");
    let model = DummyGptModel::new(&cfg, &device)?;
    println!("Model created!");

    // 输入: 假设有 5 个 token
    let token_ids = Tensor::new(&[1u32, 2, 3, 4, 5], &device)?;

    println!("\nInput tokens: [1, 2, 3, 4, 5]");
    println!("Input shape: {:?}", token_ids.shape());

    // 前向传播
    println!("\nRunning forward pass...");
    let logits = model.forward(&token_ids)?;

    println!("Output logits shape: {:?}", logits.shape());
    println!("\nLogits (first 5 vocab entries per token):");
    for i in 0..5 {
        let row = logits.get(i)?;
        let first5 = row.narrow(0, 0, 5)?;
        println!("  Token {}: {:?}", i, first5.to_vec1::<f32>()?);
    }

    Ok(())
}
