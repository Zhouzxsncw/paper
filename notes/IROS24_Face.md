# Driving Animatronic Robot Facial Expression From Speech

Boren Li<sup>˚:</sup>, Hang Li<sup>˚</sup>, Hangxin Liu<sup>:</sup> 

Abstract— Animatronic robots aim to enable natural humanrobot interaction through lifelike facial expressions. However, generating realistic, speech-synchronized robot expressions is challenging due to the complexities of facial biomechanics and responsive motion synthesis. This paper presents a principled, skinning-centric approach to drive animatronic robot facial expressions from speech. The proposed approach employs linear blend skinning (LBS) as the core representation to guide tightly integrated innovations in embodiment design and motion synthesis. LBS informs the actuation topology, enables human expression retargeting, and allows speech-driven facial motion generation. The proposed approach is capable of generating highly realistic, real-time facial expressions from speech on an animatronic face, significantly advancing robots’ ability to replicate nuanced human expressions for natural interaction. 

## I. INTRODUCTION

Accurately replicating human facial expressions is crucial for natural human-robot interaction [1–3]. Speechsynchronized, lifelike expressions enable social robots to achieve genuine emotional resonance with users [4, 5], underscoring the need to advance speech-driven expression generation for animatronic faces [6], heralding a paradigm shift in human-robot interactions. 

However, generating seamless, real-time animatronic facial expressions from speech presents two main challenges: (1) replicating the intricate biomechanics of human facial musculature [7–14], and (2) generating nuanced human expressions through responsive algorithms based on advanced imitation learning [15–19]. Overcoming these challenges necessitates a comprehensive approach that integrates embodiment design and motion synthesis. 

Conventional muscle-centric embodiment design approaches attempt to replicate human anatomy, but cloning complex biomechanics presents significant engineering obstacles. Importantly, the primary objective is achieving realistic facial skinning motions rather than internal muscle movements. Consequently, this paper proposes a novel skinningcentric embodiment design approach. 

Current methods animate robot facial skinning through 3D landmark alignment between humans and robots [15, 17]. However, relying on sparse landmarks has inherent limitations: (1) insufficient capture of expression intricacies, resulting in oversimplification; (2) coupling of facial shapes and expressions, leading to inconsistencies from varying shapes; (3) topological differences hindering viable landmark motion transfer, limiting adaptability; and (4) confinement of edits to specific landmarks, restricting semantic adjustability. Thus, a 3D landmark-free skinning representation is required. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/98ad5d93e2849ef6f1ae352bce4f52d4a386bb2b99d50f5d3b4cc9b37d74edc9.jpg)



Fig. 1: Expressive speech-driven facial dynamics achieved by the developed animatronic robot face. The figure showcases the realism and diversity of the generated robot facial expressions, synchronized with the corresponding speech input over time.


To address these challenges, this paper proposes a principled approach leveraging linear blend skinning (LBS) for embodiment design and motion synthesis. For embodiment, LBS guides a skinning-oriented actuation topology optimized for blend skinning objectives while referencing facial anatomy [20]. Tendon-driven control points actuate the skin, enabling dynamic expressions with minimal spatial constraints. For synthesis, LBS facilitates motion retargeting from human demonstrations into robot skinning references. A speech-driven model is further proposed that generates highly realistic, lip-synchronized LBS-based skinning motions in real-time through imitation learning. Significantly, the LBS-based motions are semantically editable, allowing embodiment adaptation without retraining. 

In summary, this paper presents the first principled approach for creating an animatronic robot face capable of generating expressions from speech. The proposed skinningcentric approach for embodiment design and motion synthesis significantly advances the state-of-the-art in creating dynamically expressive robot faces for natural interaction. 

## II. RELATED WORKS

Animatronic Robot Face: The evolution of animatronic robot faces can be categorized into two main phases: (1) early approaches that prioritized hardware design with preprogrammed expressions, and (2) recent studies that incorporate human motion transfer techniques. Pioneering works [7– 12] relied heavily on pre-programmed hardware and expressions, which severely limited their ability to generalize beyond a fixed set of postures. In recent years, researchers have made significant strides by integrating human motion transfer techniques, such as active appearance models [15], genetic algorithms [16], visual mimicry learning [17], Bayesian optimization [18], and MAP-Elites algorithms [19]. While these approaches have greatly enhanced the flexibility of animatronic robot faces, they remain fundamentally constrained by their reliance on mimicking observed human expressions through a master-slave mapping paradigm. Consequently, the generation of real-time, speech-driven facial expressions on animatronic platforms remains an unresolved challenge, impeding natural human-robot interaction. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/2b3b6185e8e578df40ca9cce5721a8c66f3d2347d3d3feb26924cfc8d42d3338.jpg)



Fig. 2: The proposed approach for creating a speech-driven animatronic robot face using LBS. The approach comprises three major components: (1) skinning-oriented robot development designs and constructs the animatronic face paired with a kinematics simulator based on the target skinning appearance, (2) skinning motion imitation learning involves learning an LBS-based model from 3D human demonstrations to generate facial motions from speech, and (3) speech-driven robot orchestration generates animatronic facial expressions during inference by utilizing the developed platform, simulator, and learned model.


Facial Expression Synthesis: Speech-driven facial expression synthesis has undergone significant advancements in generating photo-realistic talking head videos [21–24]. These approaches map speech to target video domains, yielding impressive visual results. More pertinent to this work are methods for speech-driven 3D facial animation that control full vertex-level facial skinning [25–29]. Although these techniques excel at creating realistic virtual renditions, they do not address the distinct challenges inherent in physical robots. This work, while related, distinguishes itself by focusing on the generation of robot facial expressions from speech on real animatronic platforms. This task presents unique difficulties that remain largely unexplored, positioning this research at the forefront of this emerging field. 

## III. PROPOSED APPROACH

## A. Approach Overview

Fig. 2 presents the proposed approach for creating a speech-driven animatronic robot face using LBS. It comprises three key components: First, skinning-oriented robot development designs and constructs the animatronic platform paired with a kinematics simulator based on the target skinning appearance. The LBS motion space is designed following blendshape design protocols [30] to enable seamless motion transfer. The actuation topology and simulator are developed concurrently to match this motion space, while the robot face is constructed considering physical constraints. Second, skinning motion imitation learning involves learning an LBS-based model from 3D human demonstrations to generate robot facial motions. Human blendshapes are carefully designed to align semantically with the robot blendshapes, ensuring consistent and expressive motion transfer. The learned model takes speech as input and outputs skinning motions as reference signals. Finally, during inference, speech-driven robot orchestration generates expressions on the animatronic face using the learned model and the developed simulator. Inverse kinematics is solved online to compute actuator commands from the generated skinning motions, enabling real-time, speech-driven expressions. 

## B. LBS Representation

LBS serves as the core representation in the proposed approach, offering several key advantages. First, it compacts intricate expressions into a small set of blendshape coefficients, facilitating human-like motions with minimal data. Second, as an expression-dependent but subject-invariant representation, LBS enables consistent performance across different faces by generating the same blendshape coefficients. This decouples embodiment design from motion synthesis, making the LBS-based motion synthesis model generalizable across diverse animatronic embodiments and semantically re-editable before retargeting, allowing adaptation without retraining. 

The robot facial skinning function $T ^ { R } ( { \pmb \theta } ) : \mathbb { R } ^ { \pmb \theta } \mapsto \mathbb { R } ^ { 3 U }$ represented using LBS is 

$$
T ^ {R} (\pmb {\theta}) = \mathbf {T} ^ {R} + B _ {E} ^ {R} (\pmb {\theta}; \mathcal {E} ^ {R}),\tag{1}
$$

where $\mathbf { T } ^ { R } \in \mathbb { R } ^ { 3 U }$ is the robot face template with U vertices, and $B _ { E } ^ { R } ( \pmb { \theta } ; \mathcal { E } ^ { R } ) : \mathbb { R } ^ { \pmb { \theta } } \mapsto \mathbb { R } ^ { 3 U }$ is the expression skinning function with B robot blendshape bases $\bar { \mathcal { E } } ^ { R }$ and coefficients θ. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/0c1c3f55e3722b499c9439ce7bd15d2400d8613308dda180bee3fadaad797e0f.jpg)



Fig. 3: The proposed skinning-oriented robot design. The figure comprises two primary components: (1) LBS-oriented kinematics design to achieve actuation topology for the facial muscular system that matches the designed LBS motion space and references facial anatomy, and (2) electro-mechanical design and development accounting for physical constraints of the embodiment, including key mechanical components of the skin, skeleton and muscular system, as well as the electrical control system.


To enable motion transfer, human blendshapes ${ \mathcal { E } } ^ { H }$ are designed to align semantically with ${ \mathcal { E } } ^ { R }$ . The human mesh topology matches that of the captured 3D human demonstrations, with the mean face shape of demonstration subjects serving as the human face template $\mathbf { T } ^ { H }$ . The human facial skinning function $T ^ { H } ( \pmb \theta ) : \mathbb { R } ^ { \pmb \theta } \mapsto \mathbb { R } ^ { 3 V }$ is 

$$
T ^ {H} (\pmb {\theta}) = \mathbf {T} ^ {H} + B _ {E} ^ {H} (\pmb {\theta}; \mathcal {E} ^ {H}),\tag{2}
$$

where V is the number of human face vertices. By transferring θ, human motions are effectively mapped to the robot. 

For the blendshape design, the Apple ARKit standard (excluding tongue-out) is adopted for its semantic meaningfulness, comprehensiveness, and interoperability. 

## IV. SKINNING-ORIENTED ROBOT DEVELOPMENT

## A. LBS-Oriented Kinematics Design

The primary objective in designing the facial muscular system is to reproduce the target LBS-based motion space rather than precisely replicating the intricate human musculature. An acceptable system only needs to match the desired skinning motions, not anatomical accuracy. 

To facilitate the design process, a kinematics simulator with adjustable topology is developed in Blender, enabling rapid computation of LBS and muscular motion spaces for iterative optimization. The design focuses on allocating skinning control points, each with bounded 6 degrees of freedom (B6DOF) for position and orientation, balancing motion flexibility and actuator complexity. 

Through iterative optimization, referencing LBS objectives and facial anatomy, the kinematics is designed. As depicted in Fig. 3, it contains a total of 21 control points: 4 for eyebrows, 4 for eyelids, 2 for eyeballs, 2 for the nose, 2 for cheeks, 6 for the mouth, and 1 for the jaw. The color-coded points show the control point locations, and lines indicate the B6DOF motion bounds. This optimized design strikes a balance between motion flexibility and system complexity. 

## B. Electro-Mechanical Design and Development

Mechanical Design: The primary objectives in the mechanical design are physically realizing the facial muscular motion space defined by the control points while accounting for physical constraints. To achieve this, a tendon-driven actuation approach is proposed, inspired by human facial anatomy. This approach enables the remote location of actuators, providing power under spatial limitations. 

Specifically, 24 actuators drive the eyes, eyebrows, nose, cheeks, and mouth, using tendons to actuate control points. Each actuator pulls or pushes a 1.5mm steel cable through a 1.5mm inner diameter Teflon conduit, enabling precise displacement control of rocker arms. Additionally, four actuators and a ball-joint linkage control the jaw module for articulation and 3D translation. The neck module, with three actuators, enables head roll, pitch, and yaw. 

The skeletal and muscular system design, shown in Fig. 3, is carefully crafted to constrain the control points’ B6DOF motions, ensuring they closely match the target LBS-based motion space. Although the facial skinning is simulated, it serves as the ultimate output of the system. To achieve the desired level of realism, a silicone mask was iteratively designed for the facial skin, taking into account factors such as shape, hardness, and thickness, inspired by human skin. The mask’s soft silicone provides exceptional flexibility, allowing the robot to achieve the desired motions with a high degree of fidelity. Experimental results validate the successful achievement of the designed robot blendshapes, serving as a compelling indicator of the effectiveness of the proposed skinning-oriented design approach. 

Electrical Design: The primary objective in the electrical design is to enable real-time skinning motion tracking by the animatronic face. To achieve responsive 25Hz control for robot orchestration, the system must actuate each control point across its full motion range within 40ms. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/c1b45598ac07d260625594ed818d443108680c5db1577faa1f0d54009fc88bfc.jpg)



Fig. 4: The proposed speech-driven facial skinning motion imitation learning method. The model (blue block) is trained through the train branch (red block) to generate blendshape coefficients from input speech by imitating human facial skinning motions. During inference (orange block), the model predicts blendshape coefficients, which are further decoded by the robot LBS decoder to generate robot facial skinning motions as reference signals for the kinematics simulator.


To meet these stringent real-time requirements, the design leverages specific components for signal processing and actuation. First, the ESP32 microcontroller is chosen for its efficient on-chip signal processing capabilities. Next, two PCA9685 boards provide interfaces for driving the total 31 servo motors required for articulation. In particular, 3 hightorque DS3115 servos are selected for the neck module to enable rapid rotations under the payload of the robot head. The 28 lower-torque LFD-01M servos optimize cost while still providing adequate speed for the finer facial motions. With a $0 . 1 2 \mathrm { s e c } / 6 0 ^ { \circ }$ at 4.8V, the servos exceed the 40ms actuation constraints when powered by a 5V lithium battery. 

The high-level system uses a GPU-accelerated PC, which generates motion references from the learned model and runs the simulator to solve inverse kinematics to generate servo commands for the ESP32 microcontroller. Experimental results validate accurate real-time tracking, demonstrating that the proposed electrical design succeeds in enabling dynamic facial expression control on the developed robot platform. 

## V. SKINNING MOTION IMITATION LEARNING

The goal is to learn a function fp¨q that maps input speech s to blendshape coefficients θ, i.e. $f ( \mathbf { s } ) \to \theta ,$ where $\pmb { \theta } \in \mathbb { R } ^ { B }$ and B is the number of blendshapes. The function $f ( \cdot )$ is learned from a dataset D containing paired examples of speech and corresponding 3D human skinning motions. The learned function should generalize to unseen speech inputs and generate realistic, expressive facial motions. 

Fig. 4 presents the proposed facial skinning motion imi tation learning method, consisting of two key branches: (1) The training branch develops a model to generate LBS-based facial skinning motions, represented by blendshape coefficients, from input speech. This model is learned from 3D human demonstrations showing linkages between speech and facial skinning motions. (2) The inference branch leverages the robot LBS decoder to transform predicted blendshape coefficients into corresponding skinning motion reference signals. These signals further drive through the robot kinematics simulator to produce lifelike facial articulation. 

## A. Model Architecture

The model architecture fp¨q takes raw speech waveforms s and generates blendshape coefficients θ representing skinning motions. It contains: (i) a frame-level speech encoder extracting embeddings via a transformer-based phoneme logit extractor and temporal fusion module, (ii) a speaking style encoder embedding conditioning vectors, and (iii) an LBS encoder projecting conditioned speech embeddings to blendshape coefficients in the range r0, 1s. 

Specifically, the phoneme extractor follows the state-ofthe-art transformer-based self-supervised pre-trained speech model, Wav2vec2 [31], finetuned on the CommonVoice dataset with 53 languages and 392 phoneme classes to enhance cross-language generalization. It outputs a phoneme logits sequence at 49Hz followed by resampling to 25Hz to match the robot orchestration rate. The temporal fusion module progressively fuses K-step neighborhood logits over a sliding window into frame-level speech embeddings, comprising $M = \log _ { 2 }$ K stacked Resnet blocks with a kernel size of 3, stride of 2, and H filters. The N-way speaking style vector accounts for cross-subject variations and is embedded with a fully-connected (FC) layer having H hidden units. The LBS encoder takes the late-fused conditioned speech embedding as input and outputs blendshape coefficients with two successive layers: an FC layer having 2H hidden units with a ReLU activation, and an FC layer with H hidden units followed by a Sigmoid activation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/0ac3517f33a22f69cd92135a91bc253c586ce9f85f75af1f3e236df295123ea3.jpg)



Fig. 5: Motion Space Validation. Actuated blendshape error for different facial regions (left figure): Color-coded skinning landmarks represent different facial regions for evaluation. MSE error distributions between simulated and physically actuated blendshapes are presented using violin plots, box and whisker plots, and scattered points, with each point representing one blendshape. Blendshapes are grouped by facial region, and only landmarks in the corresponding region are used for evaluation. Median errors are 2.41mm (eye), 3.27mm (brow), 1.76mm (nose), 4.01mm (cheek), 3.76mm (mouth), and 8.63mm (jaw). Qualitative comparison (right figure): Eight comparisons between simulated and actuated blendshapes are shown. Blendshapes (1)-(6) demonstrate high accuracy, while (7) mouth close and (8) jaw open highlight limitations in the current design, exhibiting maximum errors for their respective regions.


## B. Training

The human LBS decoder transforms blendshape coefficients into vertex displacements. It is an FC layer with zero biases and a linear activation. Its weights are set and frozen using designed blendshapes over vertex displacements. 

The loss function compares predicted and ground-truth vertices, regressing positions over the entire face. An additional weighted mouth term encourages improved lip synchronization, as mouth shapes strongly correlate with speech. For each predicted frame-level vertex, $\hat { \mathbf { y } } = \{ \hat { y } _ { i } \} _ { i \in [ 0 , V ] } .$ , the model is trained by minimizing the loss $\mathcal { L }$ compared to the ground truth vertex, $\mathbf { y } = \{ y _ { i } \} _ { i \in [ 0 , V ] }$ . The loss is 

$$
\mathcal {L} = \sum_ {i = 1} ^ {V} \left(y _ {i} - \hat {y} _ {i}\right) ^ {2} + w _ {m} \sum_ {j = 1} ^ {V _ {m}} \left(y _ {j} - \hat {y} _ {j}\right) ^ {2},\tag{3}
$$

where $V _ { m }$ is the number of vertices in the masked mouth region and $w _ { m }$ is the mouth weight. 

## C. Inference

The robot LBS decoder transforms blendshape coefficients into skinning motions using designed robot blendshapes with frozen weights. At inference, the trained model first generates a sequence of blendshape coefficients from input speech. A low-pass Butterworth filter then smooths the sequence over time for temporal stability. The filtered sequence is input to the robot LBS decoder to compute the skinning motion as the reference signal in the proposed kinematics simulator. 

## D. Implementation Details

The VOCASET dataset [26] of audio-3D scan pairs of English utterances is used for training and testing. VO-CASET was chosen for its high-quality 3D facial motion captures, diverse subjects, and phonetically balanced utterances, making it well-suited for learning speech-driven facial animations. The dataset contains 255 unique sentences, some shared across 12 subjects. VOCASET has 480 facial motion sequences captured at $6 0 \mathrm { H z } ,$ each lasts 3 to 4 seconds. The 3D face mesh has $V = 5 0 2 3$ vertices. For our purposes, VOCASET motions are resampled to 25Hz to match the robot orchestration rate. Vertex sequences are computed by subtracting the zero-pose subject-specific face meshes, then added to the designed human face template, $\mathbf { T } ^ { H }$ , to get subject-agnostic sequences. Sequences are split $8 0 / 1 0 / 1 0$ into train/validation/test sets, with two subjects reserved for testing generalizability. 

The model is implemented in PyTorch. The sliding window size K is 8 (320ms). The hidden dimension H is 64. The robot has $U = 4 7 9 2$ vertices and $B = 5 1$ blendshapes. Training uses the Adam optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9$ learning rate $1 e ^ { - 4 } .$ , weight decay $1 e ^ { - 4 } , w _ { m }$ is set to 1. A dropout layer with a dropout rate of 0.1 is added after each FC layer except the one with Sigmoid activation. 200 epochs are trained on an NVIDIA 4090 GPU („ 3sec{epoch). Wav2vec2-XLSR53 is fixed during training. For inference, an 7Hz 5th-order low-pass Butterworth filter is applied. The model generates blendshape coefficients from speech at ą 4000Hz on GPU. 

## VI. EXPERIMENTS

## A. Robot Development Experiments

The experiments in this subsection aim to validate the proposed animatronic face in achieving the designed motion space and dynamic tracking performance. These experiments provide crucial evidence for the effectiveness of the skinningoriented design approach and electro-mechanical design in enabling realistic and responsive facial expressions. 

Motion Space Validation: The first experiment quantitatively validates the physical realization of the designed LBS motion space defined by the robot blendshapes. Achieving the comprehensive set of blendshapes is essential for enabling expressive facial articulation. A VICON motion capture system is employed to track the physical skinning deformations. Reflective markers are attached to key control points and auxiliary skinning locations, corresponding to landmarks in the simulator. By individually actuating each blendshape and comparing the VICON marker positions with the simulated landmarks, the discrepancies can be measured using the mean squared error (MSE). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/bdb4275d715e862fa45dc2938ea1e68830bf28e7c76356f6e8ddd3bbcb9e3f9f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/fff454af69620bd8e085a750fb672bbf9376ae179fafd57f5e988bea1b34e15c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/ff6d31a3175961d4991ec071fc459b0d2b965274600dbc449195c45c4f67b745.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/8bf091f13f570eb000925c85acd2479e94cce22b44d90ce2b57f67c7b0f7f4f6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/5989607e175b8d267ba3c49b0c30f6bda732e19f14d57e180fed54fd24ffe90f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/5158f79ee93439f253acca29d8f596dc67b6becf8088fd47ca5e147f522bb69c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/dc42f91715cebe1d0b50f24c665ef55c331346b800f0f8498110cef4b38d125d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/214adb320ababd374fbd2a4ca28aa8bf62d5b5260e72397cb5ed295b92d8b7ee.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/40ae1b098640bce4bb2883026345d5ab153262e8869f5025455ebe15e9505ccb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/aa60c053-0ac8-497b-851e-87fed88eb28d/9c5097778b966ee933bf5dabe270dc7dbf8bae08d90abebbd4c43f14bd0ed626.jpg)



Fig. 6: Tracking Performance Validation. MSE error distributions between simulated and physically actuated facial articulation sequences are presented using violin plots, box and whisker plots, and scattered points, with each point representing one frame. Evaluation landmarks are grouped by facial region. Ten realistic facial articulation sequences from different speakers with distinct speaking styles were evaluated. Mean median errors across the ten sequences for each facial region are 2.56mm (eye), 3.39mm (brow), 1.74mm (nose), 3.08mm (cheek), 3.86mm (mouth), and 5.03mm (jaw). The results demonstrate that the animatronic robot face achieves accurate tracking performance across various facial regions and speaking styles.


The results, presented in Fig. 5, demonstrate highly accurate realization of the designed motion space. Across various facial regions, the MSE errors are consistently remain at the millimeter scale. For instance, the mouth blendshapes exhibit a median error of 3.76mm, while the eye blendshapes show a median error of 2.41mm. These low errors confirm the successful translation of the designed blendshapes into physical skinning deformations, validating the effectiveness of the skinning-oriented design approach. 

Tracking Performance Validation: The second experiment evaluates the animatronic platform’s dynamic tracking performance in real-time by assessing its ability to follow reference facial motions. Achieving responsive and accurate tracking is crucial for enabling lifelike and synchronous facial expressions. The experimental setup employs the same VICON system, with markers attached to the robot’s control points and auxiliary skinning locations. A diverse set of 10 realistic facial articulation sequences from different speakers with distinct speaking styles from the training set, enhanced with random blinks, serve as reference signals. These sequences encompass a wide range of full facial dynamics, representative of the system’s intended operating conditions. 

As demonstrated in Fig. 6, the animatronic platform exhibits remarkable tracking performance. The MSE errors between the reference signals and the tracked positions remain consistently low over time across all facial regions and speaking styles. Comparing these dynamically achieved errors with those achieved statically during the motion space validation experiment presented in Fig. 5, the errors for different facial regions are very similar, on the millimeter scale. These results validate the effectiveness of the proposed electro-mechanical design in enabling precise and responsive facial motion control. 

## B. Imitation Learning Experiments

The following experiment assesses the validity of the proposed speech-driven motion synthesis method in generating realistic and expressive robot skinning motion references. Evaluating motion quality poses challenges due to the com plex relationship between speech and facial expressions. As various plausible motions can match the same utterance, metrics such as prediction error are ineffective for assessing synthesis quality. Instead, a blind user study is employed to perform a perceptual evaluation and gauge the naturalness of the generated motions. 

The user study involves a binary comparison between test sequences serving as ground truth and the developed model’s output conditioned on all training subjects. The test sequences are distinct from the training and validation sets and involve subjects not used during training. These sequences, originally processed as dense human facial skinning, are projected onto the human blendshape basis by solving a constrained linear optimization problem. The resulting blendshape coefficients are then retargeted to the robot’s facial skinning to obtain the ground-truth motions. 

Both the ground-truth and generated robot skinning motions are rendered as videos without facial textures to ensure that the evaluation focuses solely on motion quality. Participants are asked to choose which video in each pair shows more natural motion matching the speech or exhibits similar quality. The display order of the pair is randomized to prevent selection bias. Participants must first pass a qualification test to ensure meaningful results. 

In total, 12 qualified participants evaluated 80 video pairs three times each. The results show that participants preferred the generated motions in 29.2% of the tests and the ground truth in 45.0%, while judging 25.8% of the examples to exhibit similar quality from both methods. These findings validate the effectiveness of the motion synthesis model in generating robot skinning motions from speech that are sufficiently natural and expressive to satisfy human perception. 

The overall performance, including the motion synthesis on the developed animatronic robot face, is demonstrated in the supplementary video. The developed system is capable of automatically generating appropriate and dynamic facial expressions from speech in real-time, indicating the validity of the proposed skinning-centric approach that tightly integrates embodiment design and motion synthesis. 

## VII. CONCLUSIONS AND FUTURE WORKS

This paper introduces a novel, skinning-centric approach to drive animatronic robot facial expressions from speech. By employing LBS as the core representation, tightly integrated innovations in embodiment design and motion synthesis are achieved. LBS serves as a guiding principle for the actuation topology, enables seamless human expression retargeting, and facilitates speech-driven facial motion generation. Experimental results conclusively demonstrate that the developed animatronic robot face successfully generates highly realistic facial expressions from speech automatically and in realtime, validating the effectiveness of the proposed approach. 

Future research can build upon this work in two primary directions. First, a general robot facial muscular system design approach could be explored that drives any human-like robot face simply by replacing the customized skull and skinning components, facilitating easy fabrication and control of customized animatronic faces. Second, more advanced imitation learning methods could empower speech-synchronized robot facial expressions with controllable emotions to enable complex facial signals that support richer, more engaging human-robot interactions. As animatronic robot technology continues to mature, realistic robotic faces will likely become commonplace across entertainment, education, healthcare, and other facets of life. This pioneering skinning-based approach establishes a strong foundation for that future. 

## REFERENCES



[1] T. Fong, I. Nourbakhsh, and K. Dautenhahn, “A survey of socially interactive robots,” Robot. Auton. Syst., vol. 42, pp. 143–166, 2003. 





[2] C. Breazeal, K. Dautenhahn, and T. Kanda, “Social robotics,” Springer handbook of robotics, pp. 1935–1972, 2016. 





[3] S. Saunderson and G. Nejat, “How robots influence humans: A survey of nonverbal communication in social human–robot interaction,” Int. J. Soc. Robot., vol. 11, pp. 575–608, 2019. 





[4] N. Lazzeri, D. Mazzei, M. Ben Moussa, N. Magnenat-Thalmann, and D. De Rossi, “The influence of dynamics and speech on understanding humanoid facial expressions,” IJARS, vol. 15, no. 4, 2018. 





[5] J. D. Lomas, A. Lin, S. Dikker, D. Forster, M. L. Lupetti, G. Huisman, J. Habekost, C. Beardow, P. Pandey, N. Ahmad, et al., “Resonance as a design strategy for ai and social robots,” Frontiers in neurorobotics, vol. 16, p. 850489, 2022. 





[6] J. Złotowski, D. Proudfoot, K. Yogeeswaran, and C. Bartneck, “Anthropomorphism: opportunities and challenges in human–robot inter action,” Int. J. Soc. Robot., vol. 7, pp. 347–360, 2015. 





[7] K. Berns and J. Hirth, “Control of facial expressions of the humanoid robot head roman,” in IROS, pp. 3119–3124, IEEE, 2006. 





[8] J.-H. Oh, D. Hanson, W.-S. Kim, Y. Han, J.-Y. Kim, and I.-W. Park, “Design of android type humanoid robot albert hubo,” in IROS, pp. 1428–1433, IEEE, 2006. 





[9] T. Hashimoto, S. Hitramatsu, T. Tsuji, and H. Kobayashi, “Development of the face robot saya for rich facial expressions,” in SICE-ICASE Int. Joint Conf., pp. 5423–5428, IEEE, 2006. 





[10] D. Mazzei, N. Lazzeri, D. Hanson, and D. De Rossi, “Hefes: An hybrid engine for facial expressions synthesis to control human-like androids and avatars,” in BioRob, pp. 195–200, IEEE, 2012. 





[11] C.-Y. Lin, C.-C. Huang, and L.-C. Cheng, “An expressional simplified mechanism in anthropomorphic face robot design,” Robotica, vol. 34, no. 3, pp. 652–670, 2016. 





[12] W. T. Asheber, C.-Y. Lin, and S. H. Yen, “Humanoid head face mechanism with expandable facial expressions,” IJARS, vol. 13, no. 1, p. 29, 2016. 





[13] Z. Faraj, M. Selamet, C. Morales, P. Torres, M. Hossain, B. Chen, and H. Lipson, “Facially expressive humanoid robotic face,” HardwareX, vol. 9, p. e00117, 2021. 





[14] Z. Yan, Y. Song, R. Zhou, L. Wang, Z. Wang, and Z. Dai, “Facial expression realization of humanoid robot head and strain-based anthropomorphic evaluation of robot facial expressions,” Biomimetics, vol. 9, no. 3, p. 122, 2024. 





[15] F. Ren and Z. Huang, “Automatic facial expression learning method based on humanoid robot xin-ren,” IEEE Trans. Hum.-Mach. Syst., vol. 46, no. 6, pp. 810–821, 2016. 





[16] H.-J. Hyung, H. U. Yoon, D. Choi, D.-Y. Lee, and D.-W. Lee, “Optimizing android facial expressions using genetic algorithms,” Applied Sciences, vol. 9, no. 16, p. 3379, 2019. 





[17] B. Chen, Y. Hu, L. Li, S. Cummings, and H. Lipson, “Smile like you mean it: Driving animatronic robotic face with learned models,” in ICRA, pp. 2739–2746, IEEE, 2021. 





[18] D. Yang, W. Sato, Q. Liu, T. Minato, S. Namba, and S. Nishida, “Optimizing facial expressions of an android robot effectively: a bayesian optimization approach,” in Humanoids, pp. 542–549, 2022. 





[20] U. Zarins, Anatomy of Facial Expression. Exonicus, 2017. 





[19] B. Tang, R. Cao, R. Chen, X. Chen, B. Hua, and F. Wu, “Automatic generation of robot facial expressions with preferences,” in ICRA, pp. 7606–7613, IEEE, 2023. 





[21] Y. Zhou, X. Han, E. Shechtman, J. Echevarria, E. Kalogerakis, and D. Li, “Makelttalk: speaker-aware talking-head animation,” TOG, vol. 39, no. 6, pp. 1–15, 2020. 





[22] B. Liang, Y. Pan, Z. Guo, H. Zhou, Z. Hong, X. Han, J. Han, J. Liu, E. Ding, and J. Wang, “Expressive talking head generation with granular audio-visual control,” in CVPR, pp. 3387–3396, 2022. 





[23] W. Zhang, X. Cun, X. Wang, Y. Zhang, X. Shen, Y. Guo, Y. Shan, and F. Wang, “Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation,” in CVPR, pp. 8652–8661, 2023. 





[24] J. Wang, K. Zhao, Y. Ma, S. Zhang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Facecomposer: A unified model for versatile facial content creation,” NIPS, vol. 36, 2024. 





[25] T. Karras, T. Aila, S. Laine, A. Herva, and J. Lehtinen, “Audio-driven facial animation by joint end-to-end learning of pose and emotion,” TOG, vol. 36, no. 4, pp. 1–12, 2017. 





[26] D. Cudeiro, T. Bolkart, C. Laidlaw, A. Ranjan, and M. J. Black, “Capture, learning, and synthesis of 3d speaking styles,” in CVPR, pp. 10101–10111, 2019. 





[27] A. Richard, M. Zollhöfer, Y. Wen, F. De la Torre, and Y. Sheikh, “Meshtalk: 3d face animation from speech using cross-modality disentanglement,” in ICCV, pp. 1173–1182, 2021. 





[28] Y. Fan, Z. Lin, J. Saito, W. Wang, and T. Komura, “Faceformer: Speech-driven 3d facial animation with transformers,” in CVPR, pp. 18770–18780, 2022. 





[29] R. Daneˇcek, K. Chhatre, S. Tripathi, Y. Wen, M. Black, and T. Bolkart,ˇ “Emotional speech-driven animation with content-emotion disentanglement,” in SIGGRAPH Asia, pp. 1–13, 2023. 





[30] J. P. Lewis, K. Anjyo, T. Rhee, M. Zhang, F. H. Pighin, and Z. Deng, “Practice and theory of blendshape facial models.,” Eurographics, vol. 1, no. 8, p. 2, 2014. 





[31] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” vol. 33, pp. 12449–12460, 2020. 

