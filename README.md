# 3dGoldPoseDetection

https://github.com/Molten-Ice/3dGolfPoseEstimation/assets/58090981/afa0bb36-8375-4ae2-8031-6f5b39e21b02

Note for the labeller:
- Grip keypoints are labelled in the centred of the hands colinear with the club
- The head keypoint is on the intersection where the shaft meets the head, only slightly into the head

## 2d keypoint predictions

![good-predictions](/media/amazing-predictions.png)

## 2 keypoint predictions

|                | Before training | After training |
|----------------|-----------------|----------------|
| test Evaluation| bbox: 143.12,   | bbox: 14.19,   |
|                | grip: 102.43,   | grip: 6.87,    |
|                | head: 187.06    | head: 10.71    |

## Augmented data used for training

![augmentations](/media/augmentations.png)


