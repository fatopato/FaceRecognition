#code forked and tweaked from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
#to extend, just add more people into the known_people folder

import face_recognition
import cv2
import numpy as np
import os
import glob
import logging
import datetime
 # Logging format

logging.basicConfig(filename='entrance.log', format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
logging.info("Started")
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#make array of sample pictures with encodings
known_face_encodings = []
known_face_names = []
cwd = os.getcwd()
known_path = os.path.join(cwd, 'known_people/')
unknown_path = os.path.join(cwd, 'unknown_people')
pathExists = os.path.exists(known_path)

if not pathExists:
    print("You need to create a 'known_people' folder and put the images there")

else:
    #make an array of all the saved jpg files' paths
    list_of_files = [f for f in glob.glob(known_path+'*.jpg')]
    #find number of known faces
    number_files = len(list_of_files)

    names = [n.replace(known_path, "").replace(".jpg","") for n in list_of_files.copy()]

    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        known_face_encodings.append(globals()['image_encoding_{}'.format(i)])

        # Create array of known names
        names[i] = names[i]
        known_face_names.append(names[i])

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    previous_face_encodings = np.ones((1,128))

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        height, width = frame.shape[:2]

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every othe  r frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            #print("Number of faces: ", len(face_locations))

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    logging.warning(name + " came in")

                else:
                    cv2.imwrite("unknown_people/unknown-" + datetime.datetime.now().strftime("%x-%X")+".jpg",frame)
                    logging.warning("Unknown entrance")
                face_names.append(name)
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            color = ()
            color_red = (0, 0, 255)
            color_green = (0, 255, 0)
            color_yellow = (0, 255, 255)

            love = ""
            #if name == "fatih":
            #    sub_face = frame[(top-50 if top-50 > 0 else 0) : (bottom+50 if bottom+50<height else height), (left-50 if left-50>0 else 0):(right+50 if right+50<width else width)]
            #    cv2.imwrite('fato.jpg', sub_face)
            if name == "Unknown":
                color = color_red

            elif name == "love":
                color = color_yellow
                love = "I love you"
            else:
                color = color_green

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, love, (left + 6, top + 36), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Tracker', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
