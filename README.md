# IDAnonymizationPraktikum  

## Abstract  
ID anonymization is important for protecting individual's privacy however, the challenge lies in hiding the identity while maintaining the "utility" of the image (ensuring it still looks like a realistic human with the correct expressions and background and sometimes even some partial features from the original data)  

## Motivation and Literature Review  
Most recent researches highlight two significant advancements:  
    - Reverse Personalization (2025): Removes identity-specific features and allows users to modify attributes (like age) while preserving the scene.  
    - NullFace (2026): Allows specific parts of the face (like the nose or eyes) to be kept original while the overall identity is hidden, ensuring that non-identity attributes like gaze and pose are preserved.  

## Project Aim: Privacy and Intuition  
The core aim of this project is to solve a paradox in current anonymization: to hide an ID, most systems first have to "extract" and handle that ID. This creates significant safety risks, such as memory leaks or insider attacks where the identities to be anonymized are in plain sight. And even if the environment is secure, achieving anonymization by using the original ID (what we want to NOT end up with in the results image) is somewhat corrupted as the idea. It's like a multiple choice question where wrong answers are known so the core of anonymization is never learned, but cheated by not selecting the wrong answers.  
We propose a more intuitive and secure solution. Without explicitly extracting the ID, we eliminate the risk of handling sensitive data. Our method aims to anonymize faces without ever "holding" the original identity, making the process inherently safer while allowing multi-aspect control over the level of anonymization.  

## Proposed Method: The Pipeline  
Our method introduces a specialized pipeline designed for privacy-first generation. When an image enters the system, it follows these steps:  
    1. Feature Extraction: Instead of extracting the ID, we extract two "NON-ID" features. Could be any sort of representation that can symbolize that to preserve and what to hide (preferably something fit to pass to diffusion models for conditional generation).  
    2. Dual-Path Processing:  
        * The first (more fitting) feature is used for DDIM inversion of the original image to maintain the structural context.    
        * The second feature is used to generate a completely new ID encoding. Different methods could be used to obtain an already know or entirely new ID.
    3. Controlled Generation: During the diffusion process, we use the "NON-ID" feature as a reference for "what to not look like" and the new ID encoding for "what to look like.".  

This approach achieves two breakthroughs not found in NullFace or Reverse Personalization:  
    - No ID Extraction: The sensitive identity is never isolated or stored.  
    - Total Control: Users can control the appearance of the resulting image both through general facial features and specific ID-wise adjustments, offering a superior balance of privacy and customization.  