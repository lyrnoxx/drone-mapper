import cv2
import numpy as np
import os

def demo_feature_matching():
    img_folder = "images-true"
    images = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    if len(images) < 2:
        print("Not enough images for demo")
        return

    # Load two consecutive images (known high overlap)
    img1_path = os.path.join(img_folder, images[0])
    img2_path = os.path.join(img_folder, images[1])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector (SIFT is more accurate for mapping than ORB usually)
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    print("Extracting features...")
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters for fast matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches.")

    if len(good_matches) > 10:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = gray1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw results
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        
        # Save the result
        cv2.imwrite("scripts/feature_matching_demo.png", result_img)
        print("Demo image saved to scripts/feature_matching_demo.png")
        
        # In a real application, we would use the Homography 'M' to correct the DJI GPS pose.
        # Translation in pixels -> Translation in meters (using GSD)
        tx = M[0, 2]
        ty = M[1, 2]
        print(f"Calculated Pixel Shift: x={tx:.2f}, y={ty:.2f}")
    else:
        print("Not enough matches found.")

if __name__ == "__main__":
    demo_feature_matching()
