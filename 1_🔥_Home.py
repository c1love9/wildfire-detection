import torch
import streamlit as st  # type: ignore
import requests # type: ignore
from PIL import Image
from pathlib import Path
import io
from dataset_store import dataset_root
from model_catalog import fire_models, general_models, default_model_index
from inference_ui import load_detector, show_model, with_segmentation, show_segmentation
from image_input import demo_image_choices, load_image
from ui_theme import apply_theme, page_header

PROJECT_ROOT = Path(__file__).resolve().parent

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold, device='cpu'):
    res_image, _, class_counts, latency_ms = model.predict_image(image, conf_threshold, iou_threshold)

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'
        
        if v > 1:
            prediction_text += 's'
        
        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    # Calculate inference latency
    latency = round(latency_ms / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    return res_image, prediction_text

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="auto",
        layout="wide",
    )
    
    apply_theme()
    page_header('智能火灾报警系统', '识别火焰与烟雾，从一张图片开始验证；视频监测可进一步演示连续确认、分级报警与事件回放。',
                section='IMAGE ANALYSIS', tags=('YOLO 火烟识别', 'SAM 像素分割', '报警与历史回看'))
    control_col, result_col = st.columns([1, 1.9], gap='large')
    with control_col:
        with st.container(border=True):
            st.subheader('识别设置')
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

            choices = general_models(PROJECT_ROOT) if model_type == 'General' else fire_models(PROJECT_ROOT, dataset_root(PROJECT_ROOT))
            model_files = list(choices)
            if not choices:
                st.error('No model weights found. See docs/quick_start.md.')
                st.stop()
            preferred_model = 'yolov8n' if model_type == 'General' else 'fire_n'
            default_index = default_model_index(choices, preferred_model)
    
            with col2:
                selected_model = st.selectbox("Select Model Size", model_files, index=default_index, format_func=lambda key: choices[key].label)
            choice = choices[selected_model]
            with st.expander('模型来源与训练记录'):
                show_model(choice)
            sizes = sorted({160, 320, 640, 960, 1280, choice.imgsz})
            imgsz = st.selectbox('推理图像尺寸', sizes, index=sizes.index(choice.imgsz))

            device_options = ['CPU']
            if torch.cuda.is_available():
                device_options.append('GPU (CUDA)')
            device_label = st.selectbox('Inference Device', device_options, index=len(device_options) - 1)
            device = 'cuda:0' if device_label == 'GPU (CUDA)' else 'cpu'
            if device == 'cuda:0':
                st.caption(f"Using {torch.cuda.get_device_name(0)}")

            # Model and general info
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("What is General?"):
                    st.caption("The General model is an additional model that was added for demonstration purposes.")
                    st.caption("It is pre-trained on the COCO dataset, which consists of various objects across 80 classes.")
                    st.caption("Please note that this model may not be optimized specifically for fire detection.")
                    st.caption("For accurate fire and smoke detection, it is recommended to choose the Fire Detection model.")
                    st.caption("The General model can be used to detect objects commonly found in everyday scenes.")
    
            with col2:
                with st.expander("Size Information"):
                    st.caption("Models are available in different sizes, indicated by n, s, m, and l.")
                    st.caption("- n: Nano")
                    st.caption("- s: Small")
                    st.caption("- m: Medium")
                    st.caption("- l: Large")
                    st.caption("The larger the model, the more precise the detections, but the slower the inference time.")
                    st.caption("On the other hand, smaller models are faster but may sacrifice some precision.")
                    st.caption("Choose a model based on the trade-off between speed and precision that best suits your needs.")

            # Load the selected model
            segmentation_enabled = st.checkbox('开启火烟像素分割', value=False, disabled=model_type == 'General')
            try:
                model = with_segmentation(load_detector(choice, device, imgsz), segmentation_enabled and model_type != 'General')
            except Exception as exc:
                st.error(f"Unable to load {choice.label}: {exc}")
                st.stop()

            # Add a section divider

            # Set confidence and IOU thresholds
            col1, col2 = st.columns(2)
            with col2:
                conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
                with st.expander("What is Confidence Threshold?"):
                    st.caption("The Confidence Threshold is a value between 0 and 1.")
                    st.caption("It determines the minimum confidence level required for an object detection.")
                    st.caption("If the confidence of a detected object is below this threshold, it will be ignored.")
                    st.caption("You can adjust this threshold to control the number of detected objects.")
                    st.caption("Lower values allow more detections; higher values require greater confidence.")
            with col1:
                iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
                with st.expander("What is IOU Threshold?"):
                    st.caption("This is the non-maximum suppression (NMS) IoU threshold.")
                    st.caption("Lower values suppress overlapping detections more aggressively.")
                    st.caption("This slider does not set the IoU used for evaluation against ground truth.")

            # Add a section divider

    with result_col:
        with st.container(border=True):
            st.subheader('图像检测工作区')
            # Image selection
            image = None
            image_source = st.radio("Select image source:", ("Demo Image", "Enter URL", "Upload from Computer"))
            if image_source == 'Demo Image':
                try:
                    demo_images = {item['path'].name: item for item in demo_image_choices(PROJECT_ROOT)}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    st.error(f'演示图片清单无法读取：{exc}')
                    demo_images = {}
                if demo_images:
                    selected_demo = st.selectbox('Select demo image', list(demo_images),
                                                 format_func=lambda name: demo_images[name]['label'])
                    selected_image = demo_images[selected_demo]
                    try:
                        image = load_image(selected_image['path'])
                        if selected_image['curated']:
                            st.caption('精选训练集样例用于功能演示；正式性能请查看独立测试结果。')
                            with st.expander('查看无框原图'):
                                st.image(image, caption='输入原图，检测框由当前模型现场生成。', width=1000)
                    except ValueError as exc:
                        st.error(str(exc))
                else:
                    st.info('No demo images found. Upload an image or enter its URL.')
            elif image_source == "Upload from Computer":
                # File uploader for image
                uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    try:
                        image = load_image(uploaded_file)
                    except (OSError, ValueError) as exc:
                        st.error(f'Unable to read this image: {exc}')
                else:
                    image = None

            else:
                # Input box for image URL
                url = st.text_input("Enter the image URL:")
                if url:
                    try:
                        with requests.get(url, timeout=15) as response:
                            response.raise_for_status()
                            image = load_image(io.BytesIO(response.content))
                    except (requests.exceptions.RequestException, OSError, ValueError) as e:
                        st.error(f"Error loading image from URL: {e}")
                        image = None

            if image:
                # Display the uploaded image
                with st.spinner("Detecting"):
                    try:
                        if segmentation_enabled and model_type != 'General':
                            import cv2
                            import numpy as np
                            frame_result = model.predict_frame(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), conf_threshold, iou_threshold)
                            prediction = frame_result.rgb
                            text = 'Predicted ' + ', '.join(f'{count} {label}' for label, count in frame_result.counts.items())
                            show_segmentation(frame_result, downloads=True)
                        else:
                            prediction, text = predict_image(model, image, conf_threshold, iou_threshold, device)
                    except Exception as exc:
                        st.error(f'Inference failed on {device_label}: {exc}')
                        st.stop()
                    st.image(prediction, caption="Prediction", width=1000)
                    st.success(text)
        
                prediction = Image.fromarray(prediction)

                # Create a BytesIO object to temporarily store the image data
                image_buffer = io.BytesIO()

                # Save the image to the BytesIO object in PNG format
                prediction.save(image_buffer, format='PNG')

                # Create a download button for the image
                st.download_button(
                    label='Download Prediction',
                    data=image_buffer.getvalue(),
                    file_name='prediction.png',
                    mime='image/png'
                )

        
if __name__ == "__main__":
    main()
