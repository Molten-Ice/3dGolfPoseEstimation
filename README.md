# 3dGoldPoseDetection

https://github.com/Molten-Ice/3dGolfPoseEstimation/assets/58090981/afa0bb36-8375-4ae2-8031-6f5b39e21b02

Note for the labeller:
- Grip keypoints are labelled in the centred of the hands colinear with the club
- The head keypoint is on the intersection where the shaft meets the head, only slightly into the head

![good-predictions](/media/amazing-predictions.png)

Final metrics:

test Evaluation | bbox: 14.19, grip: 6.87, head: 10.71 | bbox: [200/200], keypoints: [200/200], Time taken: 8.21s

start metrics:

test Evaluation | bbox: 143.12, grip: 102.43, head: 187.06 | bbox: [200/200], keypoints: [200/200], Time taken: 8.37s

![image1](/media/train-dataset.png)

![first-golf-club-prediction](/media/first-golf-club-prediction.png)

Augmented data used for training:

![augmentations](/media/augmentations.png)
