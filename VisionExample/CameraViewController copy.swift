//
//  Copyright (c) 2018 Google Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import AVFoundation
import CoreVideo
import MLImage
import MLKit

@objc(CameraViewController)
class CameraViewController: UIViewController {

  private var isUsingFrontCamera = true
  private var previewLayer: AVCaptureVideoPreviewLayer!
  private lazy var captureSession = AVCaptureSession()
  private lazy var sessionQueue = DispatchQueue(label: Constant.sessionQueueLabel)
  private var lastFrame: CMSampleBuffer?
    
  private var isRecording: Bool = false
  private var data = [Int: [String: Any]]()
  private var curTS = 0
    
  private lazy var previewOverlayView: UIImageView = {

    precondition(isViewLoaded)
    let previewOverlayView = UIImageView(frame: .zero)
    previewOverlayView.contentMode = UIView.ContentMode.scaleAspectFill
    previewOverlayView.translatesAutoresizingMaskIntoConstraints = false
    return previewOverlayView
  }()

  /// Initialized when one of the pose detector rows are chosen. Reset to `nil` when neither are.
  private var poseDetector: PoseDetector? = PoseDetector.poseDetector(options: AccuratePoseDetectorOptions())

  /// Initialized when a segmentation row is chosen. Reset to `nil` otherwise.
  private var segmenter: Segmenter? = Segmenter.segmenter(options: SelfieSegmenterOptions())

  // MARK: - IBOutlets
  @IBOutlet private weak var cameraView: UIView!
  @IBOutlet weak var mouthState: UILabel!
  @IBOutlet weak var leftHandState: UILabel!
  @IBOutlet weak var rightHandState: UILabel!
  @IBOutlet weak var mouthThresholdLabel: UILabel!
  @IBOutlet weak var mouthThreshold: UISlider!
    
  // MARK: - UIViewController

  override func viewDidLoad() {
    super.viewDidLoad()

    previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    setUpPreviewOverlayView()
    setUpAnnotationOverlayView()
    setUpCaptureSessionOutput()
    setUpCaptureSessionInput()
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)

    startSession()
  }

  override func viewDidDisappear(_ animated: Bool) {
    super.viewDidDisappear(animated)

    stopSession()
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    previewLayer.frame = cameraView.frame
  }

  // MARK: - IBActions

  @IBAction func switchOverlay(_ sender: UIButton) {
    if annotationOverlayView.alpha == 1 {
        annotationOverlayView.alpha = 0
        annotationOverlayViewB.alpha = 0
    }else{
        annotationOverlayView.alpha = 1
        annotationOverlayViewB.alpha = 1
    }
  }
    @IBAction func setThresholdDefault(_ sender: UIButton) {
        mouthThreshold.setValue(5, animated: true)
        mouthThresholdLabel.text = "5"
    }
    @IBAction func updateThreshold(_ sender: UISlider) {
        mouthThresholdLabel.text = "\(Int(sender.value))"
        sender.setValue(Float(Int(sender.value)), animated: false)
    }
    @IBAction func toggleRecording(_ sender: UIButton) {
        isRecording.toggle()
        
        if isRecording {
            sender.setTitle("Stop", for: .normal)
            data = [Int: [String: Any]]()
            curTS = 0
        }else{
            let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let content = (try? FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)) ?? []
            let filecount = content.filter{$0.lastPathComponent.hasSuffix(".csv")}.count
            let filename = path.appendingPathComponent("data\(filecount).csv")
            do {
                try str.joined(separator: "\n").write(to: filename, atomically: true, encoding: .utf8)
            }catch{
                print(error.localizedDescription)
            }
            sender.setTitle("Record", for: .normal)
        }
    }
    
  // MARK: On-Device Detections

  private func detectFacesOnDevice(in image: VisionImage, width: CGFloat, height: CGFloat) {
    // When performing latency tests to determine ideal detection settings, run the app in 'release'
    // mode to get accurate performance metrics.
    let options = FaceDetectorOptions()
    let faceDetector = FaceDetector.faceDetector(options: options)
    var faces: [Face]
    do {
      faces = try faceDetector.results(in: image)
    } catch let error {
      print("Failed to detect faces with error: \(error.localizedDescription).")
      self.updatePreviewOverlayViewWithLastFrame()
      return
    }
    guard !faces.isEmpty else {
     print("On-Device face detector returned no results.")
      return
    }
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      for face in faces {
        strongSelf.doFaceParsing(face)
      }
    }
  }

  private func detectPose(in image: MLImage, width: CGFloat, height: CGFloat) {
    if let poseDetector = self.poseDetector {
      var poses: [Pose]
      do {
        poses = try poseDetector.results(in: image)
      } catch let error {
        print("Failed to detect poses with error: \(error.localizedDescription).")
        self.updatePreviewOverlayViewWithLastFrame()
        return
      }
      self.updatePreviewOverlayViewWithLastFrame()
      guard !poses.isEmpty else {
       print("Pose detector returned no results.")
        return
      }
      weak var weakSelf = self
      DispatchQueue.main.sync {
        guard let strongSelf = weakSelf else {
          print("Self is nil!")
          return
        }
        // Pose detected. Currently, only single person detection is supported.
        poses.forEach { pose in
          strongSelf.doPoseParsing(pose)
        }
      }
    }
  }
}