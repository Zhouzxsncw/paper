# Causal World Modeling for Robot Control

Lin Li<sup>∗</sup> Qihang Zhang<sup>∗†</sup> Yiming Luo<sup>∗</sup> Shuai Yang Ruilin Wang Fei Han 

Mingrui Yu Zelin Gao Nan Xue Xing Zhu Yujun Shen Yinghao Xu<sup>‡</sup> 

<sup>∗</sup>Equal Contribution <sup>†</sup>Project Lead <sup>‡</sup>Corresponding Author 

This work highlights that video world modeling, alongside vision-language pre-training, establishes a fresh and independent foundation for robot learning. Intuitively, video world models provide the ability to “imagine” the near future by understanding the causality between actions and visual dynamics. Inspired by this, we introduce LingBot-VA, an autoregressive diffusion framework that learns frame prediction and policy execution simultaneously. Our model features three carefully crafted designs: (1) a shared latent space, integrating vision and action tokens, driven by a Mixture-of-Transformers (MoT) architecture, (2) a closed-loop rollout mechanism, allowing for ongoing acquisition of environmental feedback with ground-truth observations, (3) an asynchronous inference pipeline, parallelizing action prediction and motor execution to support efficient control. We evaluate our model on both simulation benchmarks and real-world scenarios, where it shows significant promise in long-horizon manipulation, data efficiency in post-training, and strong generalizability to novel configurations. The code and model are made publicly available to facilitate the community. 

Website: https://technology.robbyant.com/lingbot-va Github: https://github.com/robbyant/lingbot-va Checkpoints: https://huggingface.co/robbyant/lingbot-va 

Robbyant 

## 1 Introduction

Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic manipulation [7, 11, 12, 34], demonstrating impressive capabilities in grounding linguistic instructions into visual perceptions across diverse objects and unstructured environments. However, beneath their apparent success lies a significant challenge: representation entanglement. Most existing VLAs adopt a feedforward paradigm that maps current observations to action sequences [17, 91], requiring a single neural network to simultaneously learn visual scene understanding, physical dynamics, and motor control from a unified supervision signal. This entanglement can create a bottleneck—the model must compress heterogeneous knowledge, ranging from high-dimensional visual semantics to low-dimensional motor commands, into a shared representation space. This often leads to limited sample efficiency and suboptimal generalization. Without explicit modeling of environmental evolution [25, 26, 82], reactive policies may rely on pattern matching rather than a principled understanding of physical dynamics. 

Recent attempts to bring world modeling into robotic policies span interactive neural simulators (e.g., UniSim [86]), chunk-based video-action diffusion models (e.g., UVA [40] and UWM [97]), and offline video generators for subgoal synthesis (e.g. Gen2Act [4], Act2Goal [95]). While conceptually appealing, these approaches face three primary limitations for effective closed-loop control. First, the reactivity gap: chunk/open-loop generation often rolls out long segments without incorporating real-time feedback, making it hard to adapt to disturbances. Second, limited long-term memory: chunk-wise generation can introduce inconsistencies over long horizons when history is not persistently cached. Third, causality: bidirectional attention within a segment allows future tokens to influence past predictions, which diverges from the causal nature of physical reality where the present depends only on the past. These observations motivate an autoregressive formulation for robust closed-loop reasoning. 

We propose LingBot-VA, an autoregressive diffusion world model that addresses these limitations through a unified video-action framework. Unlike autoregressive language models that predict discrete tokens, our model operates in a continuous latent space via flow matching [46, 50], autoregressively generating chunks of video and action representations through iterative denoising. While our approach conceptually separates visual dynamics prediction and action decoding [22, 27], the key architectural insight is to interleave video and action tokens into a single autoregressive sequence. Both modalities are jointly processed through a Mixture-of-Transformers (MoT) architecture [43] with shared attention. Within this unified autoregressive generation process, latent imagination and action inference occur jointly: at each autoregressive step, the model generates predicted future visual states through iterative denoising while simultaneously decoding the corresponding actions, allowing both streams to mutually condition on one another. This integration, built upon a large-scale pretrained video diffusion backbone [79], offers several advantages: (i) Reactive AR loop: because video and action tokens form a unified sequence, each autoregressive step allows the system to recalibrate based on the latest real-world observation, enabling timely adjustments to both the predicted future and motor commands; (ii) Persistent context through KV-cache: the cached key-value pairs preserve the interleaved video-action trajectory, providing a rich context that helps mitigate temporal drift; (iii) Causal consistency: causal attention masking over the unified sequence ensures that both predicted visual states and action commands are governed by preceding states, respecting the temporal arrow of physical dynamics. By incorporating real-world observations at each step, thi formulation helps mitigate the distribution drift that often affects open-loop methods in long-horizon tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/1b9515071872a93af8c9aaebd117eb012bb8b518e635065bf36620a8f83a5c78.jpg)



Figure 1. LingBot-VA : An Autoregressive World Model for Robotic Manipulation. (1) Pretraining: LingBot-VA is pretrained on diverse in-the-wild videos and robot action data, enabling strong generalization across scenes and objects. (2) Comprehensive Evaluation: We conduct extensive experiments on real-world tasks (long-horizon, deformable objects, and precision manipulation) and simulation benchmarks, significantly outperforming state-of-the-art methods including π<sub>0.5</sub>. (3) Versatile Capabilities: Beyond policy learning, our model supports visual dynamics prediction and inverse dynamics inference from robot videos. (4) Emergent Properties: Our causal world modeling approach exhibits long-range temporal memory and strong few-shot adaptation ability.


A primary challenge in deploying large-scale autoregressive video-action models is inference latency; generating high-fidelity video tokens through iterative denoising is computationally intensive. We address this through two complementary strategies. First, we introduce Noisy History Augmentation, a training scheme that enables partial denoising at inference time. The key insight is that action decoding does not always require pixel-perfect reconstruction; instead, it can rely on robust semantic structures. By training the action decoder to predict from partially noisy latent representations, we significantly reduce the computational overhead while maintaining precise action prediction. Second, we design an asynchronous coordination pipeline that overlaps computation with execution: while the robot executes current actions, the world model predicts future visual states and plans subsequent sequences. This parallelized architecture, combined with variable chunk-size training, facilitates high-frequency closed-loop control without compromising prediction quality. 

We evaluate LingBot-VA across diverse manipulation tasks in both simulation and real-world environments. Our method demonstrates competitive performance compared to state-of-the-art VLA policies, particularly in long-horizon tasks requiring temporal consistency. Our contributions are summarized as follows: 

• Autoregressive Video-Action World Modeling: We introduce an autoregressive diffusion framework that architecturally unifies visual dynamics prediction and action inference within a single interleaved sequence while maintaining their conceptual distinction. This formulation supports persistent memory through KV cache and causal consistency via attention masking. 

• Mixture-of-Transformers Architecture with Asynchronous Execution: We design a dual-stream MoT architecture with asymmetric capacity and introduce a partial denoising strategy combined with asynchronous coordination to enable efficient robotic control. 

• Superior Long-Horizon and Precision Performance: Extensive real-world and simulation experiments demonstrate consistent state-of-the-art performance, with particularly strong improvements on long-horizon and high-precision manipulation tasks. Our method also achieves significantly improved sample efficiency and strong generalization to novel scenes and object configurations. 

## 2 Preliminary

## 2.1 Flow Matching

Flow matching [46, 50, 75] is a continuous-time generative modeling framework that learns to transform a simple source distribution $( \mathrm { e . g . }$ , Gaussian noise) to a target data distribution through a continuous flow. Given a data sample $x _ { 1 }$ and a noise sample $\epsilon \sim \mathcal { N } ( 0 , I )$ , flow matching defines a time-dependent vector field $v _ { s } : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$ that describes the instantaneous velocity of particles flowing from ϵ to x . The trajectory $x ^ { ( s ) }$ evolves according to the ordinary differential equation (ODE): 

$$
\frac {d x ^ {(s)}}{d s} = v _ {s} (x ^ {(s)}), \quad x ^ {(0)} = \epsilon \sim \mathcal {N} (0, I),\tag{1}
$$

where $s \in [ 0 , 1 ]$ denotes the flow time. 

The model is trained to predict this vector field by minimizing: 

$$
\mathcal {L} _ {\mathrm{FM}} = \mathbb {E} _ {s, \epsilon , x _ {1}} \left[ \| v _ {\theta} (x ^ {(s)}, s) - \dot {x} ^ {(s)} \| ^ {2} \right],\tag{2}
$$

where ${ \dot { x } } ^ { ( s ) }$ is the true velocity along the interpolation path, typically defined as $x ^ { ( s ) } = ( 1 - s ) \epsilon + s x _ { 1 }$ , giving ${ \dot { x } } ^ { ( s ) } = x _ { 1 } - \epsilon$ 

At inference, samples are generated by solving the learned ODE from $s = 0 \mathrm { t o } s = 1$ 

$$
x _ {1} = \epsilon + \int_ {0} ^ {1} v _ {\theta} (x ^ {(s)}, s) d s.\tag{3}
$$

## 2.2 Video Generation with Conditional Flow Matching

Recent video generation models [23, 35, 54, 79] leverage flow matching to generate videos conditioned on text or images. These models operate in the latent space of pretrained video autoencoders, where visual observations are encoded as latent representations $z _ { t } = E ( o _ { t } )$ using encoder $E \left( \mathrm { e . g . } \right.$ from video diffusion models) 

Given a conditioning signal c (text prompt or initial image), the flow matching model learns to generate a sequence of latent video frames $\mathbf { z } = \{ z _ { 1 } , \dots , z _ { T } \}$ by predicting the vector field: 

$$
v _ {\theta} (\mathbf {z} ^ {(s)}, s \mid c) = \frac {d}{d s} \mathbf {z} ^ {(s)},\tag{4}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/2a1c944d8ccbc6ed85769f3051c36c887aaea64c80a37bb0ed18b40d49f8aca0.jpg)



Figure 2. Framework overview: LingBot-VA is conditioned by autoregressive diffusion for unified video-action world modeling. We leverage a dual-stream Mixture-of-Transformers (MoT) architecture that interleaves video and action tokens within a single sequence. At each autoregressive step, the video stream (initialized from Wan2.2-5B) first predicts future latent visual states via flow matching. Then the action stream decodes corresponding actions through inverse dynamics conditioning on the predicted visual transitions.


where $s \in [ 0 , 1 ]$ is the flow time and $\mathbf { z } ^ { ( s ) }$ represents the latent video at flow step s. The generation process starts from noise $\mathbf { z } ^ { ( 0 ) } \stackrel { \cdot } { = } \epsilon \stackrel { \cdot } { \sim } \mathcal { N } ( 0 , I )$ and integrates the learned vector field to obtain the final latent video $\bar { \mathbf { z } } ^ { ( 1 ) }$ , which is then decoded to pixel space. This bidirectional generation framework enables flexible synthesis from text descriptions or seed images. 

## 3 Method

## 3.1 Problem Statement & Approach Overview

We study robotic manipulation as a sequential decision-making problem under partial observability. At each timestep $t ,$ the agent receives a visual observation $o _ { t } \in \mathcal { O }$ and executes an action $a _ { t } \in \mathcal A$ , which induces a transition in the underlying physical world and produces the next observation $o _ { t + 1 }$ 

Vision-Language-Action (VLA) Policies. Most existing VLA policies learn a direct, reactive mapping from observation history to actions: 

$$
a _ {t} \sim \pi_ {\theta} (\cdot | o _ {t}),\tag{5}
$$

through imitation learning on robot demonstration data. While this end-to-end approach has shown impressive results, it suffers from a fundamental coupling problem: the model must simultaneously learn visual scene understanding, physical dynamics, and motor control from a single supervision signal of paired observations and actions. This entanglement leads to poor sample efficiency and limited generalization, as the model struggles to disentangle visual reasoning from action prediction without explicit dynamics modeling. 

Our Approach. Unlike VLA policies that directly learn action distributions, we adopt a world modeling perspective: instead of learning $\pi ( \boldsymbol { a } _ { t } \mid \boldsymbol { o } _ { t } )$ , we predict how the visual world will evolve, then infer actions based on these predictions.· Our approach operates in two stages: 

$$
\text {(Stage 1) Visual dynamics prediction:}
$$

$$
(\text { Stage   2 }) \text { Inverse   dynamics: }
$$

$$
\begin{array}{l} o _ {t + 1} \sim p _ {\theta} (\cdot | o _ {\leq t}), \\ a _ {t} \sim g _ {\psi} (\cdot | o _ {t}, o _ {t + 1}). \end{array}\tag{6}
$$

Stage 1 learns to predict future visual observations given observation history. Stage 2 uses an inverse dynamics model to decode actions from desired visual transitions. This decomposition enables Stage 1 to leverage large-scale video data for learning physical priors, while Stage 2 only requires robot demonstrations to ground visual predictions in executable actions. 

Method Overview. Figure 2 illustrates the details of our framework. Our method consists of three key components, detailed in the following subsections: (§3.2) Autoregressive Video-Action World Modeling describes how we model visual dynamics in latent space and decode actions from predicted state transitions—this is the coreformulation of our approach; (§3.3) LingBot-VA: Unified Architecture & Training presents our unified model for video-action pretraining, including the architecture design and training objective—this is the instantiation of our formulation; (§3.4) Real-time Deployment & Asynchronous Inference introduces our deployment strategy that enables real-time control through parallelized prediction and execution—this is the practical realization for robotic control. 

## 3.2 Autoregressive Video-Action World Modeling

Previous video world models either focus on open-ended video prediction [54] or learn action-conditioned interactive environments [13, 56] primarily for game or simulation domains, which may not directly transfer to precise robotic manipulation. To leverage rich visual dynamics priors from video data for robot manipulation, we propose a unified video-action world modeling framework that jointly models visual observations and robot actions within a single autoregressive process. Unlike prior approaches that either decouple video prediction from action inference [16, 27] or rely on bidirectional diffusion within segments [97], our method unifies video and action within a single causal autoregressive framework, enabling persistent memory through KV cache and seamless integration of real-time observations. 

World Dynamics with Autoregressive Modeling. Recent world models for robotics often adopt bidirectional video generation approaches [4, 20, 24, 42] or learn interactive simulators [86], which face fundamental limitations for closed-loop control. Open-loop methods that generate entire long sequences in one shot incur prohibitive computational cost and cannot incorporate real-time feedback for error correction. Chunk-based diffusion methods that generate video segments sequentially [22, 97] suffer from two critical issues: (1) they lack persistent memory across chunks, as each chunk is generated independently without access to the full history, leading to temporal inconsistencies and drift over long horizons; (2) the bidirectional attention within each chunk violates causality, preventing seamless integration with real-time observations during execution. 

The physical world, however, is inherently causal and autoregressive: the present state depends only on the past, and we cannot observe the future before it occurs. This fundamental property motivates our autoregressive world modeling approach, which offers three critical advantages over chunk-based diffusion for robotic control: (1) Persistent Memory: by explicitly conditioning on the complete observation history through causal attention and KV cache, the model maintains long-term context and temporal coherence across the entire trajectory, avoiding the “amnesia” problem of chunk-based methods; (2) Causal Consistency: the unidirectional dependency structure naturally aligns with closed-loop execution, where new observations can be seamlessly incorporated as they arrive; (3) Efficiency: chunk-wise prediction with parallel generation within each chunk balances computational efficiency with autoregressive flexibility, enabling high-frequency control with real-time error correction. 

We formalize this as an autoregressive process: at each step, the world model predicts the next chunk of K video frames using conditional flow matching: 

$$
o _ {t + 1: t + K} \sim p _ {\theta} (\cdot | o _ {\leq t}),\tag{7}
$$

where tokens within each chunk are generated in parallel via bidirectional attention, while maintaining causal structure across chunks. This chunk-wise formulation balances generation efficiency with autoregressive flexibility for closed-loop correction. 

Video-Action State Encoding. Operating directly on pixel-level video observations is computationally prohibitive due to the high dimensionality and redundancy of raw visual data. We leverage a causal video VAE [79] to compress visual observations into compact latent tokens $z _ { t } = E ( o _ { t } \mid o _ { < t } ) \in \mathbb { R } ^ { N \times C }$ , where N is the number of spatial tokens after passing into video VAE, and $C$ is the channel number. By conditioning on previous latent states, the encoder maintains temporal coherence while processing observations sequentially, naturally aligning with our autoregressive world modeling framework. To align robot actions with visual tokens, we project action vectors to token embeddings $\boldsymbol { a } _ { t } \in \mathbb { R } ^ { D }$ via a lightweight MLP ϕ(·) where D is the dimension of the video token after patchfication, enabling unified interleaving of visual and action tokens as in prior approaches [5, 22]. 

Latent Video State Transition. While standard video generation models predict future frames based solely on visual history, robotic manipulation requires accounting for the embodiment’s physical state and interaction with the environment. During deployment, the robot’s state evolves through continuous interaction: each action modifies the embodiment’s configuration (e.g., gripper position, joint angles), which in turn influences how the scene evolves. 

In many manipulation settings, actions encode absolute pose information $( \mathrm { e . g . }$ , end-effector poses in world coordinates), so the action history $a _ { < t }$ effectively captures the trajectory of the embodiment’s configuration. Conditioning on action history thus provides knowledge of how the robot has moved and interacted with objects, consistent with prior action-conditioned video/world models [22, 86, 97]. We extend our autoregressive formulation to condition on both observation and action histories: 

$$
z _ {t + 1: t + K} \sim p _ {\theta} (\cdot | z _ {\leq t}, a _ {<   t}),\tag{8}
$$

where $z _ { t }$ is the latent visual state and $a _ { t }$ is the action token. This enables the world model to ground predictions in the embodiment’s state, ensuring that predicted observations reflect the robot’s physical interaction with the scene. 

Inverse Dynamics for Action Decoding. Once the world model predicts future visual states, we leverage these predictions to plan actions. Rather than directly predicting actions from current observations, we employ an inverse dynamics model that infers actions by conditioning on desired future observations, enabling the policy to reason about what action leads to a desired visual outcome. 

However, simply conditioning on the current and next states $( z _ { t } , z _ { t + 1 } )$ is insufficient for accurate action prediction. The action history $a _ { < t }$ encodes the embodiment’s state trajectory for determining feasible actions, while the observation history $z _ { < t }$ provides temporal context for multi-step interactions (e.g., whether an object was previously grasped). We therefore formulate inverse dynamics as: 

$$
a _ {t: t + K - 1} \sim g _ {\psi} (\cdot | \hat {z} _ {t + 1: t + K}, z _ {\leq t}, a _ {<   t}),\tag{9}
$$

where the inverse dynamics model $g _ { \psi }$ takes as input the predicted chunk of visual states $\hat { z } _ { t + 1 : t + K }$ inferred by Eq. 8, observation history $z _ { \le t }$ , and action history $a _ { < t }$ . This mirrors recent IDM-based policies [1, 20, 22, 55, 73] that leverage future targets to infer feasible actions while maintaining consistency with embodiment dynamics. 

## 3.3 LingBot-VA: Unified Architecture & Training

Architecture. To jointly model video and action generation, we leverage a dual-stream diffusion transformer architecture that performs conditional flow matching for autoregressive prediction. Our model consists of two parallel transformer backbones: a video stream initialized from Wan2.2-5B (a large-scale pretrained video generation model with dimension $d _ { v }$ [79]), and an action stream with same depth but significantly smaller width $d _ { a } \ll d _ { v }$ . This asymmetric design is motivated by the observation that action distributions are inherently simpler than visual data requiring fewer parameters to model effectively while maintaining expressive capacity for visual dynamics. 

Video Sparsification. Video frames exhibit significant temporal redundancy, especially in robotic manipulation where scenes evolve gradually. We sparsify the video sequence by temporally downsampling frames by a factor of $\tau = 4$ reducing visual tokens while improving efficiency [5]. Since actions evolve at higher frequency than visual changes, we interleave the downsampled video tokens with action tokens in temporal order: for each video frame $o _ { t } .$ , we associate τ consecutive actions $\{ a _ { t , 1 } , a _ { t , 2 } , \ldots , a _ { t , \tau } \}$ , forming a unified sequence $[ z _ { t } , a _ { t , 1 } , a _ { t , 2 } , \ldots , a _ { t , \tau } , z _ { t + 1 } , \ldots ]$ for joint modeling. This design means that predicting K video frames corresponds to generating $\tau K$ actions, enabling high-frequency control while maintaining efficient video generation. 

Mixture-of-Transformer Block. To enable interaction while preserving modality-specific feature spaces, we employ a Mixture-of-Transformers (MOT) architecture [5, 19, 43], where video and action tokens are processed by separate transformer blocks at each layer, then fused via cross-modal attention [5]. At each layer, the video and action streams independently compute their query, key, and value matrices using separate QKV projection matrices, maintaining distinct feature spaces for each modality. To align dimensions for cross-modal fusion, action tokens are first projected to the video dimension via a linear layer, participate in joint self-attention, then projected back to their original dimension via a residual connection that preserves the action-specific representations. This MOT design allows video and action to mutually influence each other through attention while maintaining separate parameterizations, preventing interference between modality-specific feature representations. For action decoding, the final action stream outputs are mapped to low-dimensional action vectors via a linear projection head. 

Action Network Initialization. Proper initialization of the action stream is critical for training stability and convergence. We find that training the action network from scratch leads to unstable optimization and slow convergence, as the action tokens’ output distribution initially diverges significantly from the video distribution, disrupting the joint attention mechanism. To address this, we initialize the action network weights by interpolating the pretrained video weights according to the action dimension, then apply a scaling factor $\alpha = \sqrt { d _ { v } / d _ { a } }$ to preserve output variance, where $d _ { v }$ and $d _ { a }$ are the video and action dimensions. This initialization strategy ensures that action tokens start with output distributions comparable to video tokens, stabilizing early-stage training and accelerating convergence. 

Variable Chunk Size Training. To enable flexible deployment, we randomly sample the chunk size K from a predefined range during training. By training with variable chunk sizes $( { \bf e . g . } , K \in [ 1 , 8 ] )$ , the model learns to generate coherent predictions across different temporal horizons. At inference time, this allows freely selecting the chunk size to balance computational efficiency and planning horizon—larger chunks reduce the number of autoregressive steps but require longer per-step computation, while smaller chunks enable more frequent closed-loop correction. In our experiments, we use $K = 4$ for deployment as a practical trade-off. 

Teacher Forcing for Unified Video-Action Training. In §3.2, we formulated both visual dynamics prediction (Eq. 7) and inverse dynamics (Eq. 8) as autoregressive modeling problems, where each prediction conditions on the history of observations and actions. This unified autoregressive formulation enables a natural training strategy: we can treat the interleaved video-action sequence as a single unified sequence and train the model using standard next-token prediction, analogous to language modeling in NLP [76]. 

Specifically, given an episode with interleaved tokens, we train the model to predict each token conditioned on all preceding tokens in the sequence. This is implemented via teacher forcing: during training, we use ground-truth tokens from the dataset as context for predicting subsequent tokens, rather than model-generated predictions. The causal dependency structure is enforced through attention masking (Figure 3)—each token can only attend to tokens that appear earlier in the temporal sequence. 

Importantly, teacher forcing is particularly well-suited for robotic manipulation: unlike pure generative modeling where it leads to train-test distribution mismatch, robot policies naturally retrieve real-world observations during deployment, directly matching the training regime. This formulation offers two key benefits: (1) unifying video and action prediction under a single training objective enables end-to-end learning of world dynamics and action inference; (2) by processing episodes in parallel with causal attention masking, we efficiently optimize both components across all timesteps in a single forward pass. 

Noisy History Augmentation. The primary bottleneck during inference remains video token generation—the number of video tokens are much larger than action tokens, and each requires multiple denoising steps through the flow matching 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/1fd1c0423c335579484952d8d6b2d4e85bee84e72c8b8cc9d1d52a2b53b14806.jpg)



Figure 3. Teacher Forcing Attention Mask: Causal attention mask for unified video-action pretraining. Each token can only attend to preceding tokens in the temporal sequence.


process. To address this, we introduce a noise augmentation strategy during training that enables partial denoising at test time. The key insight is that action prediction does not require fully denoised video representations; the inverse dynamics model can learn to extract action-relevant information from partially noisy video states. Specifically, during training, we randomly augment the video history $z _ { \le t }$ with noise following the same interpolation scheme as flow matching: 

$$
\tilde {z} _ {\leq t} = \left\{ \begin{array}{l l} (1 - s _ {\text { aug }}) \epsilon + s _ {\text { aug}} z _ {\leq t}, & p = 0. 5, \quad s _ {\text { aug }} \in [ 0. 5, 1 ],   \epsilon \sim \mathcal {N} (0, I) \\ z _ {\leq t}, & 1 - p = 0. 5 \end{array} \right.\tag{10}
$$

This augmentation trains the action decoder to predict actions from partially noisy video representations. 

At inference time, this enables a significant speedup: instead of fully denoising video tokens from $s = 0 \mathrm { ~ t o ~ } s = 1$ we only need to denoise to $s = 0 . 5$ , halving the number of denoising steps for video generation while maintaining action prediction quality. 

Algorithm 1 KV Cache Inference
Require: Initial observation $o_0$ , chunk size $K$ , KV cache $\mathcal{C}$ 1: $z_0 \leftarrow E(o_0)$ , $\mathcal{C} \leftarrow \{z_0\}$ 2: $t \leftarrow 0$ 3: loop
4: Sample $\epsilon \sim \mathcal{N}(0, I)$ ▷ Generate video chunk (integrate to $s = 0.5$ )
5: $\tilde{z}_{t+1:t+K} \leftarrow \epsilon + \int_0^{0.5} v_\theta(z_{t+1:t+K}^{(s)}, s \mid \mathcal{C}) ds$ 6: Sample $\epsilon \sim \mathcal{N}(0, I)$ ▷ Generate action chunk (integrate to $s = 1$ )
7: $a_{t:t+K-1} \leftarrow \epsilon + \int_0^1 v_\psi(a_{t:t+K-1}^{(s)}, s \mid \tilde{z}_{t:t+K}, \mathcal{C}) ds$ 8: for $i = t$ to $t + K - 1$ do
9: Execute $a_i$ , receive $o_{i+1}$ ▷ Execute and collect observations
10: $z_{i+1} \leftarrow E(o_{i+1})$ 11: end for
12: $\mathcal{C} \leftarrow \mathcal{C} \cup \{z_{t+1:t+K}, a_{t:t+K-1}\}$ ▷ Update KV cache
13: $t \leftarrow t + K$ 14: end loop 

Training Objective. We jointly optimize both video and action using flow matching with the noisy history augmentation described above. For video tokens $z _ { t }$ , the dynamics loss supervises velocity field prediction conditioned on (potentially noisy) history: 

$$
\mathcal {L} _ {\mathrm{dyn}} = \mathbb {E} _ {t, s, z _ {t + 1}, \epsilon} \left[ \| v _ {\theta} (z _ {t + 1} ^ {(s)}, s, \tilde {z} _ {\leq t}, a _ {<   t} | c) - \dot {z} _ {t + 1} ^ {(s)} \| ^ {2} \right],\tag{11}
$$

where $s \in [ 0 , 1 ]$ is flow time, $z _ { t + 1 } ^ { ( s ) } = ( 1 - s ) \epsilon + s z _ { t + 1 }$ with $\epsilon \sim \mathcal { N } ( 0 , I ) , \dot { z } _ { t + 1 } ^ { ( s ) } = z _ { t + 1 } - \epsilon , \tilde { z } _ { \le t }$ is the augmented history (Eq. 10), and c is the language instruction. For action tokens ${ { a } _ { t } } ,$ the inverse dynamics loss conditions on current and next observations: 

$$
\mathcal {L} _ {\mathrm{inv}} = \mathbb {E} _ {t, s, a _ {t}, \epsilon} \left[ \| v _ {\psi} (a _ {t} ^ {(s)}, s, \tilde {z} _ {\leq t + 1}, a _ {<   t} | c) - \dot {a} _ {t} ^ {(s)} \| ^ {2} \right],\tag{12}
$$

where $a _ { t } ^ { ( s ) } = ( 1 - s ) \epsilon + s a _ { t }$ with $\epsilon \sim \mathcal { N } ( 0 , I ) , \tilde { z } _ { t } , \tilde { z } _ { t + 1 }$ are the (potentially noisy) current and next video tokens, and c is the language instruction. The complete objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { d y n } } + \lambda \mathcal { L } _ { \mathrm { i n v } }$ 

## 3.4 Real-time Deployment & Asynchronous Inference

KV Cache for Efficient Autoregressive Inference. Our autoregressive formulation naturally enables KV cache acceleration during inference. Since each prediction step conditions on the history of observations and actions, we cache the key-value pairs from previous tokens to avoid redundant computation. At each autoregressive step, only the new tokens (current observation and predicted actions) require full attention computation, while cached history tokens are reused. Algorithm 1 describes the complete inference procedure with KV cache. 

Asynchronous Prediction and Execution. Despite the efficiency gains from KV cache and partial denoising, autoregressive prediction still incurs non-negligible latency that can violate real-time control requirements. To address this, we introduce an asynchronous inference strategy that pipelines action prediction with execution, effectively hiding prediction latency. We illustrate the difference between synchronous and asynchronous inference in Fig. 4. 

The key insight is to overlap computation with execution (Fig. 4B): While the robot executes the current action chunk $a _ { t } .$ , the model simultaneously predicts the subsequent action chunk $a _ { t + 1 }$ conditioned on the most recent real observation $z _ { t - 1 }$ (received after the execution of $a _ { t - 1 } )$ . For simplicity, we use $z _ { t }$ to denote latent observations (ignoring the video VAE compression) instead of $o _ { t }$ in this section. We discard all history data before timestamp t − 1 and use the hat notation ˆ to mark predicted visual content. Consequently, the model’s active context is limited to the executed action chunk $a _ { t - 1 }$ , the recent ground-truth observation $z _ { t - 1 }$ , the currently executing action $a _ { t } .$ , and its corresponding visual forecast $\hat { z } _ { t } . \mathrm { ~ A ~ }$ naive auto-regressive implementation (Fig. 4B-1) is to store these tokens into the KV cache and predict $\hat { z } _ { t + 1 }$ . However, we observed that such a design frequently leads to open-loop degradation and trajectory drift. Because the video generative model inherently favors temporal smoothness, it tends to "continue" the hallucinated video $\hat { z } _ { t }$ while ignoring the critical physical feedback provided by the real observation $z _ { t - 1 }$ , eventually causing the model to lose its capacity to react to the environment. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/2f1dd779d3df445e43b52eaf351350fed41058f9ed31a86a7f9e0392b5a5aaf1.jpg)



Figure 4. Asynchronous pipeline design overview: The traditional synchronous pipeline (A) suffers from delays caused by blocked computations, while the asynchronous pipeline (B) addresses this issue by enabling parallel computation and execution. However, a naive asynchronous implementation (B-1) relies on outdated visual predictions. In contrast, we improve and refine asynchronous prediction through forward dynamic prediction (B-2), which updates stale predictions with recent real-world observations.


Algorithm 2 Asynchronous Inference and Execution
Require: Initial observation $o_0$ , chunk size $K$ , KV cache $\mathcal{C}$ 1: $z_0 \leftarrow E(o_0)$ ; $\mathcal{C} \leftarrow \{z_0\}$ 2: $\tilde{z}_{1:K}, a_{0:K-1} \leftarrow \text{PREDICT}(\mathcal{C})$ ▷ Cold Start
3: ObsQueue $\leftarrow \emptyset$ ▷ Thread-safe queue for incoming real observations
4: $t \leftarrow 0$ 5: loop
6: parallel:
Branch A: Robot Execution
7: async EXECUTOR( $a_{t:t+K-1}$ , ObsQueue) ▷ Execute pre-computed actions
Branch B: Inference with FDM Grounding
8: if $t > 0$ then $o_{t-K+1:t} \leftarrow \text{ObsQueue.dequeue()}$ ▷ Get real observation $z_{t-K+1:t} \leftarrow E(o_{t-K+1:t})$ $\mathcal{C} \leftarrow \mathcal{C} \cup \{z_{t-K+1,t}, a_{t-K:t-1}\}$ ▷ Cache feedback
end if $\mathcal{C}_{\text{tmp}} \leftarrow \mathcal{C} \cup \{a_{t:t+K-1}\}$ $z_{t+1:t+K} \leftarrow \text{FDM}(\mathcal{C}_{\text{tmp}})$ $\mathcal{C}_{\text{tmp}} \leftarrow \mathcal{C}_{\text{tmp}} \cup \{z_{t+1:t+K}\}$ $\tilde{z}_{t+K+1:t+2K}, a_{t+K:t+2K-1} \leftarrow \text{PREDICT}(\mathcal{C}_{\text{tmp}})$ $t \leftarrow t + K$ 9: end loop 

To mitigate this, we introduce a Forward Dynamics Model (FDM) grounded step into our inference pipeline (Fig. 4B-2). Instead of relying on stale forecasts, we replace it by executing a forward dynamics pass: the model uses the recent feedback $z _ { t - 1 }$ and "imagines" the resulting visual state $z _ { t }$ after applying action $a _ { t }$ . By caching this feedback-grounded prediction instead of a stale forecast, we force the model to re-align with environmental feedback before predicting $z _ { t + 1 }$ This design enhances our asynchronous algorithm into a robust closed-loop system, enabling the robot to effectively perceive and react to real-world changes. 

Algorithm 2 formalizes this asynchronous pipeline. During post training, we additionally incorporate a forward dynamics prediction loss: 

$$
\mathcal {L} _ {\mathrm{fdm}} = \mathbb {E} _ {t, s, \hat {z} _ {t + 1}, \epsilon} \left[ \| v _ {\psi} (\tilde {z} _ {t + 1}, s, z _ {t}, a _ {t}, \tilde {z} _ {<   t}, \hat {a} _ {<   t} | c) - \dot {z} _ {t + 1} ^ {(s)} \| ^ {2} \right],\tag{13}
$$

## 4 Experiments

## 4.1 Dataset Curation and Preprocessing

We curate a large-scale training corpus by aggregating existing public robot manipulation datasets. All datasets undergo preprocessing to ensure consistency in data format and annotation quality, and are split into 90% training and 10% validation per dataset to monitor training dynamics. 

Unified Action Representation. To achieve cross-embodiment generalization, we define a universal action interface to adapt to different datasets. We use a dual-arm representation where each robotic arm is characterized by both end-effector pose (EEF) and joint angles. The end-effector pose consists of XYZ coordinates and a rotation quaternion (7 dimensions). For joint angles, we support a maximum of 7 degrees of freedom for single-arm embodiments; if a robot has fewer than 7 joint dimensions, we pad the missing dimensions with zeros to maintain a unified 7-dimensional representation. Each arm also has one gripper action dimension. Therefore, the total action dimensionality for dual-arm systems is: $7 _ { \mathrm { E E F } } + 7 _ { \mathrm { j o i n t s } } + 1 _ { \mathrm { g r i p p e r } }$ per arm, resulting in $( 7 + 7 + 1 ) \times 2 = 3 0$ dimensions. 

Training Data Composition. We aggregate data from six sources spanning diverse embodiments, environments, and task categories: 

• Agibot [2]: Large-scale dataset with diverse manipulation tasks from mobile manipulators. 

• RoboMind [81]: Multi-embodiment manipulation demonstrations. 

• InternData-A1 [74]: Large-scale simulation dataset for sim-to-real transfer. 

• OXE [53]: Multi-embodiment dataset; we use the OpenVLA subset. 

• UMI Data [18, 45, 48, 51, 60, 92]: Human demonstration dataset collected via universal manipulation interface<sup>1</sup>, excluding DexUMI. 

• RoboCOIN [84]: Cross-embodiment bimanual robotics data. 

In total, our training corpus comprises approximately 16K hours of robot manipulation data across diverse tasks and environments, including internally collected demonstrations. 

## 4.2 Implementation & Training Details

Implementation Details. We use Wan2.2-5B as the backbone for the video stream, with hidden dimension $d _ { v } = 3 0 7 2$ and 30 transformer layers. The action stream shares the same depth but uses a reduced hidden dimension $d _ { a } = 7 6 8$ (4× smaller), resulting in approximately 350M additional parameters and a total model size of 5.3B parameters. Both streams employ RoPE positional encoding and are connected via the MoT architecture described in §3.3. We adopt the Wan2.2 causal VAE for tokenization with $\mathbf { a } ~ 4 \times 1 6 \times 1 6$ (temporal × height × width) compression ratio, combined with a patchify operation that further reduces spatial dimensions by 2. The encoded views are concatenated along the width dimension, resulting in a total of $N = 1 9 2$ spatial tokens per frame. The action encoder ϕ and decoder are implemented as single-layer MLPs with hidden dimension 256. We normalize actions using per-dimension quantile normalization statistics computed from the training set. Task instructions are encoded using a frozen T5 text encoder [59] and injected via cross-attention. During training, chunk size K is randomly sampled from [1, 4]. 

For inference, we use Euler solver with 3 steps for video tokens (integrating to $s = 0 . 6 )$ and 10 steps for action tokens (integrating to $s = 1 . 0 )$ . Video CFG scale is set to 5.0, while action CFG scale is set to 1.0. During training, noise augmentation is applied with probability $p = 0 . 5$ and $s _ { \mathrm { a u g } } \sim \mathrm { U n i f o r m } [ 0 . 5 , 1 . 0 ]$ . Following LLM practices, we pack multiple episodes into long sequences (up to 10K tokens) with attention masks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/4704c026d157876e1a887fd453545e8014bbf5dae56cd674891c5e7bfb29650a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/894bee6f187769edd947bcc58a48de8defd2d558fad5aff1cefc112aff8c7bbd.jpg)



Figure 5. Real-world deployment results. We evaluate LingBot-VA on six manipulation tasks across three categories: longhorizon tasks (Make Breakfast, Pick Screws), precision tasks (Insert Tubes, Unpack Delivery), and deformable & articulated object manipulation (Fold Clothes, Fold Pants). Our method achieves state-of-the-art performance on both metrics.


Pre-Training Details. We pretrain LingBot-VA on the curated dataset for 1.4T tokens. We use the AdamW optimizer with peak learning rate $1 \times 1 0 ^ { - 4 }$ , weight decay 0.01, and cosine annealing schedule with linear warmup. Training is conducted in bfloat16 mixed precision with gradient clipping at 2.0. We apply classifier-free guidance with text dropout rate 0.1. The loss weight λ for inverse dynamics is set to 1. The dataset is sampled uniformly across all sources to ensure balanced learning. We monitor convergence using flow matching loss on the validation set. We use uniform SNR sampler for video model. For both video and action model, we use a uniform SNR sampler. 

Post-Training Details. While the pretrained model exhibits zero-shot generalization to seen embodiments, adapting to novel robot platforms requires a small amount of task-specific data. We find that post-training with as few as 50 demonstrations is sufficient for effective deployment. We use a reduced learning rate of $1 \times 1 0 ^ { - 5 }$ and train for 3K steps, which yields robust performance. Alternatively, a higher learning rate of $1 \times 1 0 ^ { - 4 }$ with 1K steps also produces reasonable results, though slightly inferior, offering a faster adaptation option when computational resources are limited. 

## 4.3 Main Results

## 4.3.1 Real-world Deployment

Experimental Setup. To validate the real-world effectiveness of LingBot-VA, we deploy our model on a physical robot platform and evaluate across six diverse manipulation tasks spanning three challenging categories. (1) Long-horizon Tasks: We evaluate on Make Breakfast and Unpack Delivery, which require sequential multi-step reasoning and sustained task execution over extended time horizons. (2) Precision Tasks: We test on Insert Tubes and Pick Screws, demanding accurate positioning and fine-grained motor control for successful completion. (3) Deformable Objects: We include Fold Clothes and Fold Pants, which involve manipulating non-rigid materials that present unique control challenges. The detailed task procedures are summarized in Fig. 6. These tasks are only collected with 50 real-world demos for model training. We finetune the model for 500 steps with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a sequence length of 150,000. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/688324dcdf270307fe661fbc2db1b206da86161c996f880199df23e711d061ad.jpg)



Figure 6. Detailed task progressions and key execution steps of the six real-world tasks. Each task involves a sequence of manipulation primitives, with scoring criteria detailed in Tables S2 through S4.


Results. As shown in Fig. 5, LingBot-VA consistently achieves state-of-the-art performance across all six tasks and both evaluation metrics (success rate and progress score), substantially outperforming strong baseline $\pi _ { 0 . 5 }$ 

We highlight several key observations that validate our design choices: (1) The superior performance on long horizon tasks demonstrates that our video-action world model possesses strong temporal memory capabilities. By jointly modeling video and action sequences, the model effectively maintains task context over extended horizons, enabling coherent multi-step reasoning without losing track of intermediate goals. (2) The strong results on precision tasks validate the effectiveness of our unified latent space design. By aligning video and action representations within a shared embedding space, our model achieves tighter coupling between visual perception and motor control, resulting in more accurate and fine-grained action predictions. (3) The robust performance on deformable objects highlights the value of video generation as implicit guidance. The generated video futures provide rich predictive signals about object dynamics and state transitions, which inform the action model to produce more physically plausible manipulation trajectories for challenging non-rigid materials. 

These results collectively demonstrate that our video-action world model effectively transfers to real-world deployment, exhibiting robust performance across diverse manipulation scenarios. 

## 4.3.2 Simulation Evaluation

Experimental Setup. We evaluate LingBot-VA on two widely-used simulation benchmarks: RoboTwin 2.0 [15] and LIBERO [47], covering diverse manipulation tasks across different robot embodiments. 

(1) In RoboTwin 2.0, we adopt a multi-task training setup [5] where all models are trained on 2,500 demonstrations collected in clean scenes (50 per task) plus 25,000 demonstrations from heavily randomized scenes (500 per task). We downsample the original 50 Hz video to 12.5 Hz while maintaining the action frequency at 50Hz. The model is trained for 50K steps with a learning rate of $1 \times 1 0 ^ { - 5 }$ . To facilitate a clearer comparison of performance, we categorize the 50 RoboTwin tasks according to their horizons (e.g., Place Dual Shoes has two steps, and Stack Blocks Three has three steps). The detailed horizons are listed in Tab. S1. 

(2) In LIBERO, we train our model on four LIBERO suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. Each suite contains 10 tasks with 50 demonstrations per task (500 total). Following OpenVLA [34], we filter unsuccessful demonstrations before training. The model is finetuned for 4K steps with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a sequence length of $1 \times 1 0 ^ { 5 }$ . Specifically, we report the average success rate over three random seeds, with each seed comprising 500 evaluation trials (totally 3 × 500 = 1500) for every task suite. 

Results RoboTwin 2.0 is a challenging bimanual manipulation benchmark featuring over 50 tasks that require coordinated dual-arm control. Unlike single-arm benchmarks, RoboTwin tasks demand precise synchronization between two manipulators, making it significantly more difficult for policy learning. We evaluate under both Easy (fixed initial configurations) and Hard (varied object poses and scene layouts) settings. As shown in Tab. 1, LingBot-VA achieves an average success rate of 92.9% (Easy) and 91.6% (Hard), substantially outperforming prior methods including π<sub>0</sub>, π<sub>0.5</sub>, X-VLA, and Motus. Notably, the improvement becomes more pronounced for longer-horizon tasks: at Horizon = 3, our method achieves gains of +8.2% (Easy) and +9.1% (Hard) over the second-best approach. This suggests that our autoregressive mechanism effectively maintains long-range temporal memory, enabling more robust performance as task complexity increases. 


Table 1. Evaluation on RoboTwin 2.0 Simulation (Easy vs Hard, 50 tasks). RoboTwin 2.0 is a challenging bimanual manipulation benchmark requiring coordinated dual-arm control. Easy uses fixed initial configurations while Hard involves randomized object poses and scene layouts. <sup>∗</sup> Results for X-VLA are adopted from Motus [5]. Improvements in parentheses indicat gains over the second-best method (underlined)


<table><tr><td rowspan="2">Metric</td><td colspan="2">X-VLA* [93]</td><td colspan="2"><eq>\pi_0</eq> [7]</td><td colspan="2"><eq>\pi_{0.5}</eq> [29]</td><td colspan="2">Motus [5]</td><td colspan="2">LingBot-VA (Ours)</td></tr><tr><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td></tr><tr><td>Average Horizon = 1</td><td>81.6</td><td>82.5</td><td>66.5</td><td>61.6</td><td>85.1</td><td>80.2</td><td>91.0</td><td>90.6</td><td>94.18 (+3.2)</td><td>93.56 (+3.0)</td></tr><tr><td>Average Horizon = 2</td><td>59.3</td><td>55.9</td><td>66.1</td><td>54.7</td><td>79.3</td><td>73.0</td><td>85.2</td><td>80.9</td><td>90.35 (+5.2)</td><td>86.95 (+6.1)</td></tr><tr><td>Average Horizon = 3</td><td>61.2</td><td>66.0</td><td>61.6</td><td>50.2</td><td>78.6</td><td>67.4</td><td>85.0</td><td>84.2</td><td>93.22 (+8.2)</td><td>93.28 (+9.1)</td></tr><tr><td>Average 50 Tasks</td><td>72.9</td><td>72.8</td><td>65.9</td><td>58.4</td><td>82.7</td><td>76.8</td><td>88.7</td><td>87.0</td><td>92.93 (+4.2)</td><td>91.55 (+4.6)</td></tr></table>


Table 2. Evaluation on LIBERO benchmarks. LIBERO tests manipulation across four task suites: Spatial, Object, Goal, and Long-horizon. Our method achieves new state-of-the-art on LIBERO-Object (99.6%), LIBERO-Long (98.5%), LIBERO-Spatial (98.5%), and overall average (98.5%). Baseline results are adopted from [93].


<table><tr><td rowspan="2">Methods</td><td colspan="5">LIBERO</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg</td></tr><tr><td>Octo [72]</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>Seer [73]</td><td>-</td><td>-</td><td>-</td><td>87.7</td><td>-</td></tr><tr><td>MoDE [61]</td><td>-</td><td>-</td><td>-</td><td>94.0</td><td>-</td></tr><tr><td>SuSIE [9]</td><td>-</td><td>-</td><td>-</td><td>76.3</td><td>-</td></tr><tr><td>SpatialVLA [58]</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>TraceVLA [94]</td><td>84.6</td><td>85.2</td><td>75.1</td><td>54.1</td><td>74.8</td></tr><tr><td>CoT-VLA [90]</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>81.1</td></tr><tr><td>ThinkAct [28]</td><td>88.3</td><td>91.4</td><td>87.1</td><td>70.9</td><td>84.4</td></tr><tr><td>SmolVLA [67]</td><td>93.0</td><td>94.0</td><td>91.0</td><td>77.0</td><td>88.8</td></tr><tr><td>CronusVLA [37]</td><td>97.3</td><td>99.6</td><td>96.9</td><td>94.0</td><td>97.0</td></tr><tr><td>FLOWER [62]</td><td>97.1</td><td>96.7</td><td>95.6</td><td>93.5</td><td>95.7</td></tr><tr><td>GR00T-N1 [6]</td><td>94.4</td><td>97.6</td><td>93.0</td><td>90.6</td><td>93.9</td></tr><tr><td><eq>\pi_0</eq> [7]</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.1</td></tr><tr><td><eq>\pi_0+FAST</eq> [57]</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td></tr><tr><td>OpenVLA [34]</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>OpenVLA-OFT [32]</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td>DD-VLA [44]</td><td>97.2</td><td>98.6</td><td>97.4</td><td>92.0</td><td>96.3</td></tr><tr><td>UniVLA [78]</td><td>95.4</td><td>98.8</td><td>93.6</td><td>94.0</td><td>95.4</td></tr><tr><td>X-VLA [93]</td><td>98.2</td><td>98.6</td><td>97.8</td><td>97.6</td><td>98.1</td></tr><tr><td>LingBot-VA (Ours)</td><td>98.5 ± 0.3</td><td>99.6 ± 0.3</td><td>97.2 ± 0.2</td><td>98.5 ± 0.5</td><td>98.5</td></tr></table>

We further evaluate on LIBERO benchmark (Tab. 2). On LIBERO, we obtain an average success rate of 98.5%, with particularly strong performance on LIBERO-Long (98.5%). These results establish new state-of-the-art performance in average success rates among foundational VLAs, demonstrating the effectiveness of our video-action world model for generalist robot control. 

## 4.4 Ablation

Asynchronous v.s. synchronous. We compare our asynchronous video-action generation with a synchronous baseline on RoboTwin tasks. As shown in Tab. 3, both approaches achieve comparable success rates, but our asynchronous method completes tasks 2× faster by predicting future video and action sequences while executing current actions. This validates that asynchronous generation maintains task performance while significantly improving inference efficiency. 

Pretrained LingBot-VA v.s. WAN. To validate the design choices in our video-action architecture, we conduct a controlled ablation study comparing our pretrained LingBot-VA model with WAN (Wan2.2-5B) as the initialization baseline for fine-tuning on RoboTwin tasks. Both models are fine-tuned on the same RoboTwin dataset using identical post-training procedures (50 task-specific demonstrations, learning rate $1 \times 1 0 ^ { - 5 }$ , 3K steps). 

As shown in Tab. 3, our pretrained LingBot-VA model substantially outperforms WAN fine-tuning across both Easy and Hard settings. Specifically, LingBot-VA achieves an average success rate of 92.10% (Easy) and 91.12% (Hard), while WAN fine-tuning yields significantly lower performance. This performance gap highlights the effectiveness of our joint video-action pretraining strategy, which endows the model with rich visual-motor priors that facilitate fast 


Table 3. Ablation studies on RoboTwin 2.0 (Easy). We ablate three design choices: world modeling (AR vs. bidirectional), deployment mode (async vs. sync), and pretraining (Ours vs. WAN).


<table><tr><td>Ablation</td><td>Setting</td><td><eq>Easy_{all}</eq></td><td><eq>Easy_{Horizon = 1}</eq></td><td><eq>Easy_{Horizon = 2}</eq></td><td><eq>Easy_{Horizon = 3}</eq></td></tr><tr><td>Baseline</td><td>LingBot-VA (Ours)</td><td>92.9</td><td>94.2</td><td>90.4</td><td>93.2</td></tr><tr><td rowspan="2">Deployment</td><td>FDM-grounded Async</td><td>90.4</td><td>92.5</td><td>87.7</td><td>85.6</td></tr><tr><td>Naive Async</td><td>74.3</td><td>83.3</td><td>70.3</td><td>32.9</td></tr><tr><td>Pretrain</td><td>WAN</td><td>80.6</td><td>84.9</td><td>76.3</td><td>67.6</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/89478013d02fb70a1eccaa440c1fd6ce4efeea98238f63284df171187e5eef99.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/ccca68a192312ae318a628e63ab987653acf669b621309c84c31a31877acdad0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/8ee71d7cb8a27d1ad8c7571e43b2a10ebd4927f723683422c4feae7c2a8f8a78.jpg)



Figure 7. Training dynamic comparison between different action network initialization streategy: Random initialization leads to unstable optimization (high gradient norms) and slow convergence. Although re-using video network weights stabilizes training, the resulting performance is not optimal. Our approach, which initializes by copying pretrained video weights with proper scaling, proves to be the most effective, ensuring smooth training dynamics and faster convergence.


adaptation to complex bimanual manipulation tasks. 

Action Network Initialization. Proper initialization of the action stream is critical for training stability and convergence. We compare our curated initialization strategy (Section 3.3) with naive random initialization. 

As shown in Fig. 7, random initialization from scratch exhibits volatile training dynamics with significantly slower convergence. This instability arises because action tokens’ output distribution initially diverges dramatically from the video distribution, disrupting the joint attention mechanism in our unified architecture. In contrast, our curated initialization strategy—where action network weights are initialized by interpolating pretrained video weights with a scaling factor $\alpha = \sqrt { d _ { v } / d _ { a } }$ —produces smooth convergence and substantially lower loss. 

## 4.5 Analysis

## 4.5.1 Sample Efficiency

We investigate the data efficiency by exploring how LingBot-VA performs with limited post-training data compared to $\pi _ { 0 . 5 }$ . We conduct this evaluation on both real-world and simulation settings: the “Make Breakfast” long-horizon task and RoboTwin 2.0 Easy benchmarks, allowing us to assess data efficiency across diverse manipulation scenarios. 

As shown in Fig. 8, our method consistently outperforms π<sub>0.5</sub> across all data regimes on both real-world and simulation tasks. In the low-data regime (10 demonstrations), LingBot-VA achieves 15.6% higher progress score than $\pi _ { 0 . 5 }$ on the “Make Breakfast” task and 10.3% higher on RoboTwin 2.0 Easy, demonstrating superior sample efficiency. These results demonstrate that our method learns more effectively from limited data across diverse manipulation scenarios. 

We attribute this superior data efficiency to our video-action world model design. The jointly pretrained video generation backbone provides rich visual priors about physical dynamics and object interactions, which serve as implicit regularization during post-training. This allows the action model to leverage the world knowledge encoded in the video stream, effectively reducing the sample complexity required for adapting to new tasks. In contrast, VLA models like $\pi _ { 0 . 5 }$ lack explicit modeling of visual dynamics and thus have no structured dynamics priors to guide learning, requiring more demonstrations to learn task-specific behaviors from scratch. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/dc33717ab56e59856e98ab7eb46aae4325301d43e620f153e29602b95f50b674.jpg)



(a) Robotwin Easy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/53c0e3ca671db68af5ae5c53543a9a31696131603700d06d1b520451047faf10.jpg)



(b) Real-world



Figure 8. Sample efficiency comparison. LingBot-VA consistently outperforms π<sub>0.5</sub> across various data regimes on the “Make Breakfast” task, demonstrating superior data efficiency in the post-training stage.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/87cc7aab5843ccbca3570f9b94065a5ccf5c442255e855dfef91370fd4dee64d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/035cc5a4ef5fa03ea4bfccf3ba61b0903ee6188557c2deace411d83f75193921.jpg)



Figure 9. Temporal memory evaluation. Left: Success rates on two memory tasks (Wipe Plate and Search Box). LingBot-VA significantly outperforms π<sub>0.5</sub> on both tasks, demonstrating superior temporal state tracking ability. Right: Visualization of evaluation environments.


## 4.5.2 Temporal Memory

We design the following tasks that explicitly require maintaining state information across time to evaluate our model’s temporal memory capabilities, as shown in Figure 9. 

1. Wipe Plate—the robot must wipe a plate exactly six times, requiring it to count and remember repeated actions. 

2. Search Box—Two boxes (left and right) are in the scene, with only one containing a block. The robot opens them sequentially from right to left. In data collection, the block is equally likely to be in either box; at test time, it is always in the left box. Without memory, after finding the right box empty, the model has a 50% chance of re-opening it. With memory, it proceeds to search the left box. 

As shown in Fig. 9(a), LingBot-VA substantially outperforms $\pi _ { 0 . 5 }$ on both memory tasks. We attribute this to the autoregressive nature of our world model: during training, teacher forcing conditions predictions on full history; during inference, KV-cache naturally preserves all historical information for persistent memory. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/285ead76-3017-4764-8c05-29c3e8aa3267/88bb8bd6a654a6375ed480c9291a89834d3052ca046f7df83a461428f34b7b4f.jpg)



Figure 10. Novel object and spatial generalization. LingBot-VA successfully generalizes to objects with varying shapes, textures, and positions.


## 4.5.3 Generalization

We evaluate generalization along two axes: 

1. Novel Object Generalization—trained on pick-and-place with a single object, tested on different objects with varying shapes and textures; 

2. Spatial Generalization—trained with fixed object positions in a localized region(denoted as in-distribution (ID)), tested on random placements especially in out-of-distribution (OOD) regions. 

As shown in Fig. 10, our method demonstrates a stronger generalization in both both novel object and the out-ofdistribution position. The world model learns transferable visual representations through video prediction, capturing object-agnostic physical priors that transfer to novel scenarios. 

## 5 Related Work

Vision-Language-Action Policies. Recent advancements in Embodied AI have witnessed a paradigm shift toward large scale Vision-Language-Action (VLA) policies. By leveraging web-scale knowledge and diverse robot demonstrations, models such as π [29], GR-3 [39], and GR00T-N1 [6] achieve remarkable generalizability across various manipulation tasks without relying on hand-crafted rules, modular priors, or restricted action abstractions, enabling a more direct and expressive end-to-end mapping from perception to control. These policies typically employ pre-trained Vision Language Models (VLMs) as foundational backbones [6, 7, 11, 29, 34, 39, 87, 93], which provide superior cross-modal understanding and more generalizable action distributions compared to task-specific imitation policies like ACT [91] or Diffusion Policy [17]. Efforts have been further devoted to improving the deployability through lightweight backbones [49, 62, 67], efficient tokenization [57], real-time inference [8, 10, 70], or fine-tuning schemes [30, 32, 38]. However, despite their prowess in semantic reasoning, a fundamental limitation persists: the pre-training objectives and data distributions of standard VLMs largely overlook the fine-grained system dynamics and low-level trajectories essential for precision manipulation. While supervised fine-tuning on expensively collected large-scale robot datasets allows these models to approximate the marginal action distribution [3, 31, 53], they remain deficient in capturing the underlying transition dynamics—specifically, how the physical state of the environment should evolve and will evolve. Furthermore, most current VLA methods formulate control as a purely reactive mapping from instantaneous observations to actions. This approach inherently fails to account for the historical context necessary to resolve ambiguities in non-Markovian environments. Additionally, the static image-text pre-training inherent in VLMs fails to instill essential temporal priors. Even when augmented with memory modules [37, 65, 68], such models remain unable to reason about the causal and sequential nature of physical interactions. To address these shortcomings, recent research has pivoted towards generalist robot policies grounded in world models and generative video modeling [1, 5, 40, 64, 97]. However, these methods typically generate predictions with bidirectional attention, which violates the causal structure of physical dynamics and lacks persistent long-term memory across the full execution history. Our LingBot-VA unifies autoregressive video prediction with action decoding under a strict causal temporal structure, where each prediction conditions exclusively on past observations and actions. By maintaining a persistent KV cache over the complete interaction history, LingBot-VA ensures long-range temporal consistency and allows the policy to synchronize physical execution with the predicted visual evolution of the environment. 

World Models for Robotic Control. Inspired by human reliance on intuitive physics to anticipate environmental changes, world models aim to facilitate effective planning by predicting future dynamics. Existing approaches are generally categorized into three groups based on their state representations. The first category operates in latent space [36, 41, 63, 80], encoding task-relevant features into compact vectors to predict evolution via probabilistic [36, 52, 83] or deterministic methods [63, 85]. The second category utilizes 3D point clouds [66, 69, 77], leveraging Graph Neural Networks (GNNs) to predict geometric evolution [88, 89], which is particularly effective for manipulating deformable objects [77, 89]. The third category focuses on 2D pixel space, directly predicting future keyframes or video sequences [21, 33, 95, 96]. Our work aligns with this third category. Within this domain, approaches range from co-training with video generation for representation learning [14, 40, 97] to serving as simulators for policy learning or evaluation [71]. Our research specifically targets methods that predict future frames during execution to condition action generation. However, prior video-conditioned methods predominantly rely on open-loop generation [21, 95], presenting two significant challenges. First, the misalignment between generated videos and real-world dynamics, coupled with cumulative drift from execution errors, often leads to suboptimal performance. Second, the computational intensity of video generation imposes high latency, severely hindering real-time inference. Our method leverages KV Cache and causal masking to continuously update the model’s memory with real-world observations. This effectively transitions the system to a closed-loop control mechanism, mitigating error accumulation in long-horizon tasks. Furthermore, we introduce a partial denoising strategy, enabling action generation from intermediate representations without waiting for fully denoised frames. 

## 6 Conclusion

We present LingBot-VA, an autoregressive diffusion framework that unifies video dynamics prediction and action inference for robotic manipulation. By interleaving video and action tokens within a Mixture-of-Transformers architecture, our model captures the causal structure of physical interactions while enabling closed-loop control through continuous integration of real-world observations. Extensive evaluation demonstrates strong performance across simulation benchmarks (92.0% on RoboTwin 2.0, 98.5% on LIBERO) and real-world deployment, achieving over 20% improvement on challenging tasks compared to $\pi _ { 0 . 5 }$ with only 50 demonstrations for adaptation. These results suggest that autoregressive video-action world modeling provides a principled foundation for learning generalizable manipulation policies, offering a compelling alternative to reactive VLA paradigms. 

Future Work. Future directions include developing more efficient video compression schemes to reduce computational overhead, and incorporating multi-modal sensory inputs (tactile, force, audio) for more robust manipulation in tasks with complex contact dynamics. 

Acknowledgment. We thank Kecheng Zheng for insightful discussions and Wei Wu for valuable assistance with dataset preparation. We also thank Fangyi Xu and Yishu Shen for their help with the post-training data collection. 

## References



[1] 1X Technologies. 1x world model: From video to action. https://www.1x.tech/discover/world-model-self-learning, 2025. Accessed 2026-01-18. 





[2] AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, Shu Jiang, Yuxin Jiang, Cheng Jing, Hongyang Li, et al. Agibot world colosseo: A large-scal manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025. 





[3] Jose Barreiros, Andrew Beaulieu, Aditya Bhat, Rick Cory, Eric Cousineau, Hongkai Dai, Ching-Hsin Fang, Kunimatsu Hashimoto, Muhammad Zubair Irshad, Masha Itkina, et al. A careful examination of large behavior models for multitask dexterous manipulation. arXiv preprint arXiv:2507.05331, 2025. 





[4] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei 





Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. In Conference on Robot Learning (CoRL), 2024. 





[5] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, et al. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025. 





[6] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Jim Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 





[7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, et al. π : A vision language-action flow model for general robot control. In Robotics: Science and Systems, 2025. 





[8] Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. arXiv preprint arXiv:2506.07339, 2025. 





[9] Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pretrained image-editing diffusion models. In Int. Conf. Learn. Represent., 2024. 





[10] Kevin Black, Allen Z Ren, Michael Equi, and Sergey Levine. Training-time action conditioning for efficient real-time chunking. arXiv preprint arXiv:2512.05964, 2025. 





[11] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023. 





[12] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, et al. Rt-1: Robotics transformer for real-world control at scale. In Robotics: Science and Systems, 2023. 





[13] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, et al. Genie: Generative interactive environments. In Int. Conf. Mach. Learn., 2024. 





[14] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2502.14420, 2025. 





[15] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025. 





[16] Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. In Int. Conf. Comput. Vis., 2025. 





[17] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems, 2023. 





[18] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Robotics: Science and Systems, 2024. 





[19] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 





[20] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In Adv. Neural Inform. Process. Syst., 2023. 





[21] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023. 





[22] Yao Feng, Hengkai Tan, Xinyi Mao, Guodong Liu, Shuhe Huang, Chendong Xiang, Hang Su, and Jun Zhu. Vidar: Embodied video diffusion model for generalist bimanual manipulation. arXiv preprint arXiv:2507.12898, 2025. 





[23] Google DeepMind. Veo: A text-to-video generation system. Google DeepMind Technical Report, 2025. 





[24] Yanjiang Guo, Yucheng Hu, Jianke Zhang, Yen-Jen Wang, Xiaoyu Chen, Chaochao Lu, and Jianyu Chen. Prediction with action: Visual policy learning via joint denoising process. In Adv. Neural Inform. Process. Syst., 2024. 





[25] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 2025. 





[26] Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In Int. Conf. Learn. Represent., 2024. 





[27] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. In Int. Conf. Mach. Learn., 2025. 





[28] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Vision-language-action reasoning via reinforced visual latent planning. In Adv. Neural Inform. Process. Syst., 2025. 





[29] Physical Intelligence et al. π : A generalist robot policy with flow matching and world models. In Conference on Robot Learning (CoRL), 2025. 





[30] Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao, Zhenyu Wei, Yunhui Liu, Zhiwu Lu, and Mingyu Ding. Mixture of horizons in action chunking. arXiv preprint arXiv:2511.19433, 2025. 





[31] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, et al. Droid: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems, 2024. 





[32] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 





[33] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026. 





[34] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, et al. Openvla: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024. 





[35] Kuaishou. Kling ai. https://klingai.kuaishou.com/, 2024. 





[36] Chenchang Li, Zihao Ai, Tong Wu, Xiaosa Li, Wenbo Ding, and Huazhe Xu. Deformnet: Latent space modeling and dynamics prediction for deformable object manipulation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 14770–14776. IEEE, 2024. 





[37] Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, et al. Cronusvla: Transferring latent motion across time for multi-frame prediction in manipulation. arXiv preprint arXiv:2506.19816, 2025. 





[38] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025. 





[39] Jiacheng Li, Mengzhou Sun, Bowen Zhang, Zhe Zhao, Xiu Liu, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025. 





[40] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. In Robotics: Science and Systems, 2025. 





[41] Yunzhu Li, Jiajun Wu, Jun-Yan Zhu, Joshua B Tenenbaum, Antonio Torralba, and Russ Tedrake. Propagation networks for model-based control under partial observation. In 2019 International Conference on Robotics and Automation (ICRA), pages 1205–1211. IEEE, 2019. 





[42] Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, and Carl Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. In Conference on Robot Learning (CoRL), 2024. 





[43] Weixin Liang, LILI YU, Liang Luo, Srini Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. Transactions on Machine Learning Research, 2025. 





[44] Zhixuan Liang, Yizhuo Li, Tianshuo Yang, Chengyue Wu, Sitong Mao, Tian Nian, Liuao Pei, Shunbo Zhou, Xiaokang Yang, Jiangmiao Pang, Yao Mu, and Ping Luo. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. arXiv preprint arXiv:2508.20072, 2025. 





[45] Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao. Data scaling laws in imitation learning for robotic manipulation. In Int. Conf. Learn. Represent., 2025. 





[46] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Int. Conf. Learn. Represent., 2023. 





[47] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledg transfer for lifelong robot learning. In Adv. Neural Inform. Process. Syst., 2023. 





[48] Fangchen Liu, Chuanyu Li, Yihua Qin, Austin Shaw, Jing Xu, Pieter Abbeel, and Rui Chen. Vitamin: Learning contact-rich tasks through robot-free visuo-tactile manipulation interface. arXiv preprint arXiv:2504.06156, 2025. 





[49] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: A diffusion foundation model for bimanual manipulation. In Int. Conf. Learn. Represent., 2025. 





[50] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In Int. Conf. Learn. Represent., 2023. 





[51] Zeyi Liu, Cheng Chi, Eric Cousineau, Naveen Kuppuswamy, Benjamin Burchfiel, and Shuran Song. Maniwav: Learning robot manipulation from in-the-wild audio-visual data. arXiv preprint arXiv:2406.19464, 2024. 





[52] Bethany Lusch, J Nathan Kutz, and Steven L Brunton. Deep learning for universal linear embeddings of nonlinear dynamics. Nature communications, 9(1):4950, 2018. 





[53] Open X-Embodiment Collaboration. Open x-embodiment: Robotic learning datasets and rt-x models. In IEEE International Conference on Robotics and Automation (ICRA), 2024. 





[54] OpenAI. Video generation models as world simulators. OpenAI Technical Report, 2024. 





[55] Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond vlas. arXiv preprint 2512.15692, 2025. 





[56] Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev, Shangbang Long, et al. Genie 2: A large-scale foundation world model. https://deepmind.google/discover/blog/ genie-2-a-large-scale-foundation-world-model/, 2024. 





[57] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. In Robotics: Science and Systems, 2025. 





[58] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model. In Robotics: Science and Systems, 2025. 





[59] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67, 2020 





[60] Omar Rayyan, John Abanes, Mahmoud Hafez, Anthony Tzes, and Fares Abu-Dakka. Mv-umi: A scalable multi-view interface for cross-embodiment learning. arXiv preprint arXiv:2509.18757, 2025. 





[61] Moritz Reuss, Jyothish Pari, Pulkit Agrawal, and Rudolf Lioutikov. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning. In Int. Conf. Learn. Represent., 2025. 





[62] Moritz Reuss, Hongyi Zhou, Marcel Rühle, Ömer Erdinç Yagmurlu, Fabian Otto, and Rudolf Lioutikov. Flower: Democratizing˘ generalist robot policies with efficient vision-language-action flow policies. In Conference on Robot Learning (CoRL), 2025. 





[63] Bokui Shen, Zhenyu Jiang, Christopher Choy, Silvio Savarese, Leonidas J Guibas, Anima Anandkumar, and Yuke Zhu. Action conditional implicit visual dynamics for deformable object manipulation. The International Journal of Robotics Research, 43(4):437–455, 2024. 





[64] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. arXiv preprint arXiv:2512.06963, 2025. 





[65] Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation. arXiv preprint arXiv:2508.19236, 2025. 





[66] Haochen Shi, Huazhe Xu, Zhiao Huang, Yunzhu Li, and Jiajun Wu. Robocraft: Learning to see, simulate, and shape elasto plastic objects in 3d with graph networks. The International Journal ofRobotics Research, 43(4):533–549, 2024. 





[67] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025. 





[68] Ajay Sridhar, Jennifer Pan, Satvik Sharma, and Chelsea Finn. Memer: Scaling up memory for robot control via experience retrieval. arXiv preprint arXiv:2510.20328, 2025. 





[69] Deborah Sulsky, Shi-Jian Zhou, and Howard L Schreyer. Application of a particle-in-cell method to solid mechanics. Computer physics communications, 87(1-2):236–252, 1995. 





[70] Jiaming Tang, Yufei Sun, Yilong Zhao, Shang Yang, Yujun Lin, Zhuoyang Zhang, James Hou, Yao Lu, Zhijian Liu, and Song Han. Vlash: Real-time vlas via future-state-aware asynchronous inference. arXiv preprint arXiv:2512.01031, 2025. 





[71] Gemini Robotics Team, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, Sean Kirmani, Fangchen Liu, Anirudha Majumdar, et al. Evaluating gemini robotics policies in a veo world simulator. arXiv preprint arXiv:2512.10675, 2025. 





[72] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems, 2024. 





[73] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Seer: Predictive inverse dynamics models are scalable learners for robotic manipulation. In Int. Conf. Learn. Represent., 2025 





[74] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, et al. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy. arXiv preprint arXiv:2511.16651, 2025. 





[75] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machin Learning Research, 2024. 





[76] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Adv. Neural Inform. Process. Syst., 2017. 





[77] Yixuan Wang, Yunzhu Li, Katherine Driggs-Campbell, Li Fei-Fei, and Jiajun Wu. Dynamic-resolution model learning for object pile manipulation. arXiv preprint arXiv:2306.16700, 2023. 





[78] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025. 





[79] WanTeam. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 





[80] Manuel Watter, Jost Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. Advances in neural information processing systems, 28, 2015. 





[81] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, Shichao Fan, Xinhua Wang, Fei Liao, Zhen Zhao, Guangyu Li, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. In Robotics: Science and Systems, 2025. 





[82] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on Robot Learning (CoRL), 2022. 





[83] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on robot learning, pages 2226–2240. PMLR, 2023. 





[84] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025. 





[85] Zhenjia Xu, Jiajun Wu, Andy Zeng, Joshua B Tenenbaum, and Shuran Song. Densephysnet: Learning dense physical object representations via multi-step dynamic interactions. arXiv preprint arXiv:1906.03853, 2019. 





[86] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Pack Kaelbling, Dale Schuurmans, and Pieter Abbeel. Unisim: Learning interactive real-world simulators. In Int. Conf. Learn. Represent., 2024. 





[87] Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation. arXiv preprint arXiv:2507.17520, 2025. 





[88] Kaifeng Zhang, Baoyu Li, Kris Hauser, and Yunzhu Li. Adaptigraph: Material-adaptive graph-based neural dynamics for robotic manipulation. arXiv preprint arXiv:2407.07889, 2024. 





[89] Kaifeng Zhang, Baoyu Li, Kris Hauser, and Yunzhu Li. Particle-grid neural dynamics for learning deformable object models from rgb-d videos. arXiv preprint arXiv:2506.15680, 2025. 





[90] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025. 





[91] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems, 2023. 





[92] Zhaxizhuoma, Kehui Liu, Chuyue Guan, Zhongjie Jia, Ziniu Wu, Xin Liu, Tianyu Wang, Shuai Liang, Pengan Chen, Pingrui Zhang, Haoming Song, Delin Qu, Dong Wang, Zhigang Wang, Nieqing Cao, et al. Fastumi: A scalable and hardwareindependent universal manipulation interface. arXiv preprint arXiv:2409.19499, 2024. 





[93] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025. 





[94] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 





[95] Pengfei Zhou, Liliang Chen, Shengcong Chen, Di Chen, Wenzhi Zhao, Rongjun Jin, Guanghui Ren, and Jianlan Luo. Act2goal: From world model to general goal-conditioned policy. arXiv preprint arXiv:2512.23541, 2025. 





[96] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024. 





[97] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. In Robotics: Science and Systems, 2025. 



## Appendix

## A Real-world Evaluation Details

We present detailed evaluation results for all real-world manipulation tasks. Each task is evaluated with 20 trials for both our method and the baseline $( \pi _ { 0 . 5 } )$ . To ensure fair comparison, we adopt an alternating evaluation protocol: one trial with $\pi _ { 0 . 5 }$ , followed by one trial with our method, and so on. 

For each trial, we record the success status of every intermediate step. If a step requires a retry to succeed, we assign a score of 0.5; if it fails, the score is 0; if it succeeds on the first attempt, the score is 1. A trial is marked as successful only if all steps are completed (i.e., the total score equals the maximum possible score). 

We report two metrics: 

• Progress Score (PS): The average score across all trials divided by the maximum possible score, expressed as a percentage: PS = Average Progress × 100%. Max Steps 

• Success Rate (SR): The number of successful trials divided by the total number of trials, expressed as a percentage: $\begin{array} { r } { \mathrm { S R } = \frac { \# \mathrm { S u c c e s s f u l T r i a l s } } { N } \times 1 0 0 \% } \end{array}$ 

We evaluate on six diverse real-world tasks: Make Breakfast (10 steps: preparing a complete breakfast including toasting bread, pouring water, and plating), Pick Screws (5 steps: picking up paper, pouring screws, and inserting three screws), Fold Clothes (6 steps: folding a shirt including sleeves and smoothing), Unpack Delivery (5 steps: opening a package using a utility knife), Insert Tubes (2 categories: grasping and inserting 3 tubes), and Fold Pants (3 steps: folding pants and placing them). These tasks span long-horizon sequential manipulation, precision control, and deformable object handling. The following tables present per-trial results for each task. 


Table S1. Evaluation on RoboTwin 2.0 Simulation (Easy vs Hard, 50 tasks). RoboTwin 2.0 is a challenging bimanual manipulation benchmark requiring coordinated dual-arm control. Easy uses fixed initial configurations while Hard involves randomized object poses and scene layouts.


<table><tr><td rowspan="2">Simulation Task</td><td rowspan="2">Horizon</td><td colspan="2">Ours</td><td colspan="2"><eq>\pi_0</eq> [7]</td><td colspan="2"><eq>\pi_{0.5}</eq> [7]</td><td colspan="2">X-VLA [93]</td><td colspan="2">Motus [5]</td></tr><tr><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td></tr><tr><td>Adjust Bottle</td><td>1</td><td>90%</td><td>94%</td><td>99%</td><td>95%</td><td>100%</td><td>99%</td><td>100%</td><td>99%</td><td>89%</td><td>93%</td></tr><tr><td>Beat Block Hammer</td><td>1</td><td>96%</td><td>98%</td><td>79%</td><td>84%</td><td>96%</td><td>93%</td><td>92%</td><td>88%</td><td>95%</td><td>88%</td></tr><tr><td>Blocks Ranking RGB</td><td>3</td><td>99%</td><td>98%</td><td>80%</td><td>63%</td><td>92%</td><td>85%</td><td>83%</td><td>83%</td><td>99%</td><td>97%</td></tr><tr><td>Blocks Ranking Size</td><td>3</td><td>94%</td><td>96%</td><td>14%</td><td>5%</td><td>49%</td><td>26%</td><td>67%</td><td>74%</td><td>75%</td><td>63%</td></tr><tr><td>Click Alarmclock</td><td>1</td><td>99%</td><td>100%</td><td>77%</td><td>68%</td><td>98%</td><td>89%</td><td>99%</td><td>99%</td><td>100%</td><td>100%</td></tr><tr><td>Click Bell</td><td>1</td><td>100%</td><td>100%</td><td>71%</td><td>48%</td><td>99%</td><td>66%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Dump Bin Bigbin</td><td>1</td><td>89%</td><td>96%</td><td>88%</td><td>83%</td><td>92%</td><td>97%</td><td>79%</td><td>77%</td><td>95%</td><td>91%</td></tr><tr><td>Grab Roller</td><td>1</td><td>100%</td><td>100%</td><td>98%</td><td>94%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Handover Block</td><td>2</td><td>99%</td><td>78%</td><td>47%</td><td>31%</td><td>66%</td><td>57%</td><td>73%</td><td>37%</td><td>86%</td><td>73%</td></tr><tr><td>Handover Mic</td><td>2</td><td>94%</td><td>96%</td><td>97%</td><td>97%</td><td>98%</td><td>97%</td><td>0%</td><td>0%</td><td>78%</td><td>63%</td></tr><tr><td>Hanging Mug</td><td>2</td><td>40%</td><td>28%</td><td>14%</td><td>11%</td><td>18%</td><td>17%</td><td>23%</td><td>27%</td><td>38%</td><td>38%</td></tr><tr><td>Lift Pot</td><td>1</td><td>100%</td><td>99%</td><td>80%</td><td>72%</td><td>96%</td><td>85%</td><td>99%</td><td>100%</td><td>96%</td><td>99%</td></tr><tr><td>Move Can Pot</td><td>1</td><td>94%</td><td>97%</td><td>68%</td><td>48%</td><td>51%</td><td>55%</td><td>89%</td><td>86%</td><td>34%</td><td>74%</td></tr><tr><td>Move Pillbottle Pad</td><td>1</td><td>99%</td><td>99%</td><td>67%</td><td>46%</td><td>84%</td><td>61%</td><td>73%</td><td>71%</td><td>93%</td><td>96%</td></tr><tr><td>Move Playingcard Away</td><td>1</td><td>100%</td><td>99%</td><td>74%</td><td>65%</td><td>96%</td><td>84%</td><td>93%</td><td>98%</td><td>100%</td><td>96%</td></tr><tr><td>Move Stapler Pad</td><td>1</td><td>91%</td><td>79%</td><td>41%</td><td>24%</td><td>56%</td><td>42%</td><td>78%</td><td>73%</td><td>83%</td><td>85%</td></tr><tr><td>Open Laptop</td><td>1</td><td>92%</td><td>94%</td><td>71%</td><td>81%</td><td>90%</td><td>96%</td><td>93%</td><td>100%</td><td>95%</td><td>91%</td></tr><tr><td>Open Microwave</td><td>1</td><td>82%</td><td>86%</td><td>4%</td><td>32%</td><td>34%</td><td>77%</td><td>79%</td><td>71%</td><td>95%</td><td>91%</td></tr><tr><td>Pick Diverse Bottles</td><td>2</td><td>89%</td><td>82%</td><td>69%</td><td>31%</td><td>81%</td><td>71%</td><td>58%</td><td>36%</td><td>90%</td><td>91%</td></tr><tr><td>Pick Dual Bottles</td><td>2</td><td>100%</td><td>99%</td><td>59%</td><td>37%</td><td>93%</td><td>63%</td><td>47%</td><td>36%</td><td>96%</td><td>90%</td></tr><tr><td>Place A2B Left</td><td>1</td><td>97%</td><td>93%</td><td>43%</td><td>47%</td><td>87%</td><td>82%</td><td>48%</td><td>49%</td><td>82%</td><td>79%</td></tr><tr><td>Place A2B Right</td><td>1</td><td>97%</td><td>95%</td><td>39%</td><td>34%</td><td>87%</td><td>84%</td><td>36%</td><td>36%</td><td>90%</td><td>87%</td></tr><tr><td>Place Bread Basket</td><td>1</td><td>97%</td><td>95%</td><td>62%</td><td>46%</td><td>77%</td><td>64%</td><td>81%</td><td>71%</td><td>91%</td><td>94%</td></tr><tr><td>Place Bread Skillet</td><td>2</td><td>95%</td><td>90%</td><td>66%</td><td>49%</td><td>85%</td><td>66%</td><td>77%</td><td>67%</td><td>86%</td><td>83%</td></tr><tr><td>Place Burger Fries</td><td>2</td><td>97%</td><td>95%</td><td>81%</td><td>76%</td><td>94%</td><td>87%</td><td>94%</td><td>94%</td><td>98%</td><td>98%</td></tr><tr><td>Place Can Basket</td><td>2</td><td>81%</td><td>84%</td><td>55%</td><td>46%</td><td>62%</td><td>62%</td><td>49%</td><td>52%</td><td>81%</td><td>76%</td></tr><tr><td>Place Cans Plasticbox</td><td>2</td><td>100%</td><td>99%</td><td>63%</td><td>45%</td><td>94%</td><td>84%</td><td>97%</td><td>98%</td><td>98%</td><td>94%</td></tr><tr><td>Place Container Plate</td><td>1</td><td>99%</td><td>97%</td><td>97%</td><td>92%</td><td>99%</td><td>95%</td><td>97%</td><td>95%</td><td>98%</td><td>99%</td></tr><tr><td>Place Dual Shoes</td><td>2</td><td>94%</td><td>89%</td><td>59%</td><td>51%</td><td>75%</td><td>75%</td><td>79%</td><td>88%</td><td>93%</td><td>87%</td></tr><tr><td>Place Empty Cup</td><td>1</td><td>100%</td><td>100%</td><td>91%</td><td>85%</td><td>100%</td><td>99%</td><td>100%</td><td>98%</td><td>99%</td><td>98%</td></tr><tr><td>Place Fan</td><td>1</td><td>99%</td><td>93%</td><td>66%</td><td>71%</td><td>87%</td><td>85%</td><td>80%</td><td>75%</td><td>91%</td><td>87%</td></tr><tr><td>Place Mouse Pad</td><td>1</td><td>93%</td><td>96%</td><td>20%</td><td>20%</td><td>60%</td><td>39%</td><td>70%</td><td>70%</td><td>66%</td><td>68%</td></tr><tr><td>Place Object Basket</td><td>2</td><td>91%</td><td>88%</td><td>67%</td><td>70%</td><td>80%</td><td>76%</td><td>44%</td><td>39%</td><td>81%</td><td>87%</td></tr><tr><td>Place Object Scale</td><td>1</td><td>96%</td><td>95%</td><td>57%</td><td>52%</td><td>86%</td><td>80%</td><td>52%</td><td>74%</td><td>88%</td><td>85%</td></tr><tr><td>Place Object Stand</td><td>1</td><td>99%</td><td>96%</td><td>82%</td><td>68%</td><td>91%</td><td>85%</td><td>86%</td><td>88%</td><td>98%</td><td>97%</td></tr><tr><td>Place Phone Stand</td><td>1</td><td>97%</td><td>97%</td><td>49%</td><td>53%</td><td>81%</td><td>81%</td><td>88%</td><td>87%</td><td>87%</td><td>86%</td></tr><tr><td>Place Shoe</td><td>1</td><td>98%</td><td>98%</td><td>76%</td><td>76%</td><td>92%</td><td>93%</td><td>96%</td><td>95%</td><td>99%</td><td>97%</td></tr><tr><td>Press Stapler</td><td>1</td><td>85%</td><td>82%</td><td>44%</td><td>37%</td><td>87%</td><td>83%</td><td>92%</td><td>98%</td><td>93%</td><td>98%</td></tr><tr><td>Put Bottles Dustbin</td><td>3</td><td>87%</td><td>91%</td><td>65%</td><td>56%</td><td>84%</td><td>79%</td><td>74%</td><td>77%</td><td>81%</td><td>79%</td></tr><tr><td>Put Object Cabinet</td><td>2</td><td>85%</td><td>87%</td><td>73%</td><td>60%</td><td>80%</td><td>79%</td><td>46%</td><td>48%</td><td>88%</td><td>71%</td></tr><tr><td>Rotate QRcode</td><td>1</td><td>96%</td><td>91%</td><td>74%</td><td>70%</td><td>89%</td><td>87%</td><td>34%</td><td>33%</td><td>89%</td><td>73%</td></tr><tr><td>Scan Object</td><td>2</td><td>96%</td><td>91%</td><td>55%</td><td>42%</td><td>72%</td><td>65%</td><td>14%</td><td>36%</td><td>67%</td><td>66%</td></tr><tr><td>Shake Bottle Horizontally</td><td>1</td><td>100%</td><td>99%</td><td>98%</td><td>92%</td><td>99%</td><td>99%</td><td>100%</td><td>100%</td><td>100%</td><td>98%</td></tr><tr><td>Shake Bottle</td><td>1</td><td>100%</td><td>97%</td><td>94%</td><td>91%</td><td>99%</td><td>97%</td><td>99%</td><td>100%</td><td>100%</td><td>97%</td></tr><tr><td>Stack Blocks Three</td><td>3</td><td>99%</td><td>98%</td><td>72%</td><td>52%</td><td>91%</td><td>76%</td><td>6%</td><td>10%</td><td>91%</td><td>95%</td></tr><tr><td>Stack Blocks Two</td><td>2</td><td>100%</td><td>98%</td><td>93%</td><td>79%</td><td>97%</td><td>100%</td><td>92%</td><td>87%</td><td>100%</td><td>98%</td></tr><tr><td>Stack Bowls Three</td><td>3</td><td>86%</td><td>83%</td><td>77%</td><td>75%</td><td>77%</td><td>71%</td><td>76%</td><td>86%</td><td>79%</td><td>87%</td></tr><tr><td>Stack Bowls Two</td><td>2</td><td>94%</td><td>98%</td><td>94%</td><td>95%</td><td>95%</td><td>96%</td><td>96%</td><td>93%</td><td>98%</td><td>98%</td></tr><tr><td>Stamp Seal</td><td>1</td><td>96%</td><td>97%</td><td>46%</td><td>33%</td><td>79%</td><td>55%</td><td>76%</td><td>82%</td><td>93%</td><td>92%</td></tr><tr><td>Turn Switch</td><td>1</td><td>44%</td><td>45%</td><td>41%</td><td>42%</td><td>62%</td><td>54%</td><td>40%</td><td>61%</td><td>84%</td><td>78%</td></tr><tr><td>Average (%)</td><td></td><td>- 92.93</td><td>91.55</td><td>65.92</td><td>58.40</td><td>82.74</td><td>76.76</td><td>72.80</td><td>72.84</td><td>88.66</td><td>87.02</td></tr></table>


Table S2. Detailed evaluation results for Make Breakfast task (10 steps, max score 10).


<table><tr><td>Trial</td><td>Succ.</td><td>Grasp Plate</td><td>Grasp Bread</td><td>Grasp Fork</td><td>Place Bread</td><td>Press Toaster</td><td>Grasp Cup</td><td>Grasp Kettle</td><td>Pour</td><td>Grasp Apple</td><td>Serve</td><td>Prog.</td></tr><tr><td colspan="13">Ours</td></tr><tr><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>9</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>4</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>5</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>6</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>7</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>9</td></tr><tr><td>8</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>9.5</td></tr><tr><td>9</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>9</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>11</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>9</td></tr><tr><td>12</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>13</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>9</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>15</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>16</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>17</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>9.5</td></tr><tr><td>18</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>19</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td>20</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>10</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.75</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.90</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.90</td><td>0.95</td><td>0.95</td><td>9.70</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="8">97.0% (= 9.70/10 × 100%)</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="8">75.0% (= 15/20 × 100%)</td></tr><tr><td colspan="13"><eq>\pi_{0.5}</eq></td></tr><tr><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>9</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>9</td></tr><tr><td>4</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>8</td></tr><tr><td>5</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td><td>7</td></tr><tr><td>6</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>9</td></tr><tr><td>7</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>8</td></tr><tr><td>8</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>9</td></tr><tr><td>9</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>5</td></tr><tr><td>10</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4</td></tr><tr><td>11</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>8</td></tr><tr><td>12</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>7</td></tr><tr><td>13</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>7</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>8</td></tr><tr><td>15</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>8</td></tr><tr><td>16</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>8</td></tr><tr><td>17</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>9</td></tr><tr><td>18</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>8</td></tr><tr><td>19</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>6</td></tr><tr><td>20</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>5</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.70</td><td>1.00</td><td>1.00</td><td>0.80</td><td>0.75</td><td>0.55</td><td>0.35</td><td>0.80</td><td>0.50</td><td>0.80</td><td>0.75</td><td>7.30</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="8">73.0% (= 7.30/10 × 100%)</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="8">70.0% (= 14/20 × 100%)</td></tr></table>


Table S3. Detailed evaluation results for Pick Screws task (5 steps, max score 5).


<table><tr><td>Trial</td><td>Success</td><td>Grab Paper</td><td>Pour Screws</td><td>Screw 1</td><td>Screw 2</td><td>Screw 3</td><td>Progress</td></tr><tr><td colspan="8">Ours</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>1</td><td>4</td></tr><tr><td>2</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0.5</td><td>3.5</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>4</td></tr><tr><td>4</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>5</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>4.5</td></tr><tr><td>6</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>0.5</td><td>4</td></tr><tr><td>7</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>8</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>9</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>11</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>12</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>4</td></tr><tr><td>13</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>4</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>4</td></tr><tr><td>15</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0.5</td><td>1</td><td>2.5</td></tr><tr><td>16</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>17</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>4.5</td></tr><tr><td>18</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>1</td><td>4</td></tr><tr><td>19</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0.5</td><td>3.5</td></tr><tr><td>20</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.70</td><td>1.00</td><td>0.75</td><td>0.78</td><td>0.83</td><td>0.78</td><td>4.13</td></tr><tr><td></td><td></td><td colspan="5">82.5% (= 4.13/5 × 100%)</td></tr><tr><td></td><td></td><td colspan="5">70.0% (= 14/20 × 100%)</td></tr><tr><td colspan="8"><eq>\pi_{0.5}</eq></td></tr><tr><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>4</td></tr><tr><td>2</td><td>0</td><td>0.5</td><td>0</td><td>1</td><td>1</td><td>0.5</td><td>3</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>4</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0.5</td><td>0.5</td><td>3</td></tr><tr><td>5</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>0.5</td><td>4</td></tr><tr><td>6</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>4.5</td></tr><tr><td>7</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>8</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>1</td><td>4</td></tr><tr><td>9</td><td>0</td><td>1</td><td>0</td><td>0.5</td><td>0.5</td><td>0</td><td>2</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>11</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>1</td><td>4.5</td></tr><tr><td>12</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0.5</td><td>3.5</td></tr><tr><td>13</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0.5</td><td>3.5</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>1</td><td>4.5</td></tr><tr><td>15</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>4</td></tr><tr><td>16</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>17</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0.5</td><td>0.5</td><td>2</td></tr><tr><td>18</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>19</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0.5</td><td>0.5</td><td>3</td></tr><tr><td>20</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>4.5</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.50</td><td>0.88</td><td>0.60</td><td>0.83</td><td>0.75</td><td>0.65</td><td>3.70</td></tr><tr><td></td><td></td><td colspan="5">74.0% (= 3.70/5 × 100%)</td></tr><tr><td></td><td></td><td colspan="5">50.0% (= 10/20 × 100%)</td></tr></table>


Table S4. Detailed evaluation results for Fold Clothes task (6 steps, max score 6).


<table><tr><td>Trial</td><td>Success</td><td>Fold Half</td><td>Left Sleeve</td><td>Right Sleeve</td><td>Fold Again</td><td>Flatten</td><td>Place</td><td>Progress</td></tr><tr><td colspan="9">Ours</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>2</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>5</td></tr><tr><td>4</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>5</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>6</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>7</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>8</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>5</td></tr><tr><td>9</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>11</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>12</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>13</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>14</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>5</td></tr><tr><td>15</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>16</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>4.5</td></tr><tr><td>17</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>18</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>0</td><td>3.5</td></tr><tr><td>19</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>20</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.35</td><td>0.73</td><td>0.55</td><td>0.50</td><td>0.48</td><td>0.38</td><td>0.30</td><td>2.93</td></tr><tr><td></td><td></td><td></td><td colspan="2">48.8% (= 2.93/6 × 100%)</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="2">35.0% (= 7/20 × 100%)</td><td></td><td></td><td></td></tr><tr><td colspan="9"><eq>\pi_{0.5}</eq></td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>0.5</td><td>5</td></tr><tr><td>4</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>0</td><td>4.5</td></tr><tr><td>5</td><td>0</td><td>0.5</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>0</td><td>4</td></tr><tr><td>6</td><td>0</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>1</td><td>0</td><td>4.5</td></tr><tr><td>7</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>8</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>9</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>5</td></tr><tr><td>10</td><td>0</td><td>0.5</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>0</td><td>3</td></tr><tr><td>11</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td>12</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>5</td></tr><tr><td>13</td><td>0</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>2.5</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>15</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>1</td><td>5.5</td></tr><tr><td>16</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>0</td><td>3.5</td></tr><tr><td>17</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>6</td></tr><tr><td>18</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>0</td><td>4.5</td></tr><tr><td>19</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>20</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>3</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.30</td><td>0.83</td><td>0.80</td><td>0.75</td><td>0.53</td><td>0.60</td><td>0.28</td><td>3.78</td></tr><tr><td></td><td></td><td></td><td colspan="2">62.9% (= 3.78/6 × 100%)</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="2">30.0% (= 6/20 × 100%)</td><td></td><td></td><td></td></tr></table>


Table S5. Detailed evaluation results for Unpack Delivery task (5 steps, max score 5).


<table><tr><td>Trial</td><td>Success</td><td>Grab Knife</td><td>Push Blade</td><td>Handover</td><td>Cut Seal</td><td>Open Lid</td><td>Progress</td></tr><tr><td colspan="8">Ours</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>4</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>5</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>6</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>7</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>8</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>9</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>11</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>12</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>13</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>15</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>16</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>17</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>4.5</td></tr><tr><td>18</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>19</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>20</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.65</td><td>1.00</td><td>0.95</td><td>0.95</td><td>0.68</td><td>0.65</td><td>4.23</td></tr><tr><td></td><td></td><td colspan="5">84.5% (= 4.23/5 × 100%)</td></tr><tr><td></td><td></td><td colspan="5">65.0% (= 13/20 × 100%)</td></tr><tr><td colspan="8"><eq>\pi_{0.5}</eq></td></tr><tr><td>1</td><td>0</td><td>1</td><td>1</td><td>0.5</td><td>0.5</td><td>0</td><td>3</td></tr><tr><td>2</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>3</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>4</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>5</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>6</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>7</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>8</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>4.5</td></tr><tr><td>9</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>11</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>12</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>13</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td>14</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>15</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td><td>4.5</td></tr><tr><td>16</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>17</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>18</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td>19</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>0</td><td>3.5</td></tr><tr><td>20</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>5</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.25</td><td>1.00</td><td>1.00</td><td>0.98</td><td>0.43</td><td>0.25</td><td>3.65</td></tr><tr><td></td><td></td><td colspan="5">73.0% (= 3.65/5 × 100%)</td></tr><tr><td></td><td></td><td colspan="5">25.0% (= 5/20 × 100%)</td></tr></table>


Table S6. Detailed evaluation results for Insert Tubes task (2 categories: Grasp and Insert, max score 6).


<table><tr><td>Trial</td><td>Success</td><td>Grasp (3)</td><td>Insert (3)</td><td>Progress</td></tr><tr><td colspan="5">Ours</td></tr><tr><td>1</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>2</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>3</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>4</td><td>0</td><td>2</td><td>2</td><td>4</td></tr><tr><td>5</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>6</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>7</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>8</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>9</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>10</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>11</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>12</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>13</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>14</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>15</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>16</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>17</td><td>0</td><td>2</td><td>2</td><td>4</td></tr><tr><td>18</td><td>0</td><td>2</td><td>2</td><td>4</td></tr><tr><td>19</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>20</td><td>0</td><td>2</td><td>2</td><td>4</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.40</td><td>2.80</td><td>2.35</td><td>5.15</td></tr><tr><td></td><td colspan="2">85.8% (= 5.15/6 × 100%)</td><td></td></tr><tr><td></td><td colspan="2">40.0% (= 8/20 × 100%)</td><td></td></tr><tr><td colspan="5"><eq>\pi_{0.5}</eq></td></tr><tr><td>1</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>2</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>3</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>4</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>5</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>6</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>7</td><td>0</td><td>3</td><td>1</td><td>4</td></tr><tr><td>8</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>9</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>10</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>11</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>12</td><td>0</td><td>2</td><td>1</td><td>3</td></tr><tr><td>13</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>14</td><td>0</td><td>2</td><td>2</td><td>4</td></tr><tr><td>15</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>16</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>17</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>18</td><td>0</td><td>2</td><td>2</td><td>4</td></tr><tr><td>19</td><td>1</td><td>3</td><td>3</td><td>6</td></tr><tr><td>20</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td rowspan="3">Avg Progress Score Success Rate</td><td>0.30</td><td>2.85</td><td>1.90</td><td>4.75</td></tr><tr><td></td><td colspan="2">79.2% (= 4.75/6 × 100%)</td><td></td></tr><tr><td></td><td colspan="2">30.0% (= 6/20 × 100%)</td><td></td></tr></table>


Table S7. Detailed evaluation results for Fold Pants task (3 steps, max score 3).


<table><tr><td>Trial</td><td>Success</td><td>Fold 1</td><td>Fold 2</td><td>Place</td><td>Progress</td></tr><tr><td colspan="6">Ours</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>5</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>6</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>7</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>8</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>9</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>10</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>11</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>12</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>13</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>14</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>15</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>16</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>17</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>18</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>19</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>20</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Avg</td><td>0.70</td><td>0.90</td><td>0.70</td><td>0.70</td><td>2.30</td></tr><tr><td>Progress Score</td><td></td><td colspan="4">76.7% (= 2.30/3 × 100%)</td></tr><tr><td>Success Rate</td><td></td><td colspan="4">70.0% (= 14/20 × 100%)</td></tr><tr><td colspan="6"><eq>\pi_{0.5}</eq></td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>3</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>6</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>7</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>8</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>9</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>11</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>12</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>13</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr><tr><td>14</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>15</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>16</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>17</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>18</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>19</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>20</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Avg</td><td>0.30</td><td>0.30</td><td>0.30</td><td>0.30</td><td>0.90</td></tr><tr><td>Progress Score</td><td></td><td colspan="4">30.0% (= 0.90/3 × 100%)</td></tr><tr><td>Success Rate</td><td></td><td colspan="4">30.0% (= 6/20 × 100%)</td></tr></table>