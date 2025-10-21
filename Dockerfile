FROM palroboticssl/tiago_tutorials:noetic

RUN apt-get update

RUN echo "source /opt/ros/noetic/setup.bash " >> ~/.bashrc

RUN cd /tiago_public_ws && catkin build


RUN python3 -m pip install pip --upgrade

RUN python3 -m pip install open3d==0.10.0 
RUN python3 -m pip install ultralytics 
RUN python3 -m pip install ultralytics timm onnx onnxruntime
RUN python3 -m pip install onnxruntime-gpu
RUN python3 -m pip install onnxsim
RUN python3 -m pip install segment_anything transformers mediapipe flask

RUN python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
    model_id='IDEA-Research/grounding-dino-base'; \
    AutoProcessor.from_pretrained(model_id); \
    AutoModelForZeroShotObjectDetection.from_pretrained(model_id)"
    
RUN echo "cd /tiago_public_ws && catkin build tiago_project" >> ~/.bashrc

RUN echo "source /tiago_public_ws/devel/setup.bash" >> ~/.bashrc

RUN python3 -m pip install openai