import cv2  # opencv
from skimage.metrics import structural_similarity


# Four hash comments are implementations for activating text message service
# # # # from twilio.rest import Client
# # # # from twilio_creds import ACCOUNT_SID, AUTH_TOKEN, TWILIO_PHONE
# # # # from phone_number import PHONE_NUMBER

def ssim(A, B):
    """Takes in two images A and B"""
    
    return structural_similarity(A,B, data_range=A.max()-A.min())  # data_range is dynamically computed


# # # # client = Client(ACCOUNT_SID, AUTH_TOKEN)

# open webcam (import video)
filePath = 'e:\\Eskills-Academy-projects\\security-camera-source-code\\security-camera\\'
fileName = "video.mp4"

videoFile = filePath + fileName
videoCapture = cv2.VideoCapture(videoFile)
# webcam: videoCapture = cv2.VideoCapture(0)

# read first frame
_, currentFrame = videoCapture.read()

# convert to grey scale
currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

#initialize other frame
previousFrame = currentFrame

imageTolerance = 0.9   # lower means more tolerance
frameCompareFrequency = 10
frameCounter = 1

# # # # isFirstMessage = True
# main loop
while True:
    # Read in the next valide frame (and convert it to grey scale)
    _, currentFrame = videoCapture.read()
    if currentFrame is None:
        break
    currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

    if frameCounter % frameCompareFrequency == 0:
    # Compare two frames
        ssim_index = ssim(currentFrame, previousFrame)
        if ssim_index < imageTolerance:    #### and isFirstMessage
            # # # # client.messages.create(body='Intruder Alert!', from_=TWILIO_PHONE, to=PHONE)
            print("Intruder Alert !")
            # # # # isFirstMessage = False

        # Updating previous frame
        previousFrame = currentFrame

    # Display the video/webcam feed
    cv2.imshow('app', currentFrame)

    delay = 1
    cv2.waitKey(delay)
    frameCounter += 1


videoCapture.release()
cv2.destroyAllWindows()


