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
  private let detectors: [Detector] = [
    .onDeviceFace,
    .onDeviceText,
    .onDeviceTextChinese,
    .onDeviceTextDevanagari,
    .onDeviceTextJapanese,
    .onDeviceTextKorean,
    .onDeviceBarcode,
    .onDeviceImageLabel,
    .onDeviceImageLabelsCustom,
    .onDeviceObjectProminentNoClassifier,
    .onDeviceObjectProminentWithClassifier,
    .onDeviceObjectMultipleNoClassifier,
    .onDeviceObjectMultipleWithClassifier,
    .onDeviceObjectCustomProminentNoClassifier,
    .onDeviceObjectCustomProminentWithClassifier,
    .onDeviceObjectCustomMultipleNoClassifier,
    .onDeviceObjectCustomMultipleWithClassifier,
    .pose,
    .poseAccurate,
    .segmentationSelfie,
  ]

  private var currentDetector: Detector = .onDeviceFace
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

  private lazy var annotationOverlayView: UIView = {
    precondition(isViewLoaded)
    let annotationOverlayView = UIView(frame: .zero)
    annotationOverlayView.translatesAutoresizingMaskIntoConstraints = false
    return annotationOverlayView
  }()
    
  private lazy var annotationOverlayViewB: UIView = {
      precondition(isViewLoaded)
      let annotationOverlayView = UIView(frame: .zero)
      annotationOverlayView.translatesAutoresizingMaskIntoConstraints = false
      return annotationOverlayView
    }()

  /// Initialized when one of the pose detector rows are chosen. Reset to `nil` when neither are.
  private var poseDetector: PoseDetector? = PoseDetector.poseDetector(options: AccuratePoseDetectorOptions())

  /// Initialized when a segmentation row is chosen. Reset to `nil` otherwise.
    private var segmenter: Segmenter? = Segmenter.segmenter(options: SelfieSegmenterOptions())
    
  /// The detector mode with which detection was most recently run. Only used on the video output
  /// queue. Useful for inferring when to reset detector instances which use a conventional
  /// lifecyle paradigm.
  private var lastDetector: Detector?

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
            var str: [String] = ["Timestamp,Mouth Distance of Opening,Mouth Movement,Left Shoulder Coordinate,Left Wrist Coordinate,Left Elbow Coordinate,Left Hand Movement,Right Shoudler Coordinate,Right Wrist Coordinate,Right Elbow Coordinate,Right Hand Movement"]
            let sortedKey = Array(data.keys).sorted(by: <)
            
            for t in sortedKey {
                if
                    let item = data[t],
                    let m = item["mouthMovement"],
                    let md = item["mouthOpenness"],
                    let l = item["leftHandMovement"],
                    let ls = item["leftShoulderCoor"],
                    let lw = item["leftWristCoor"],
                    let le = item["leftElbowCoor"],
                    let r = item["rightHandMovement"],
                    let rs = item["rightShoulderCoor"],
                    let rw = item["rightWristCoor"],
                    let re = item["rightElbowCoor"] {
                    str.append("\(t),\(md),\(m),\(ls),\(lw),\(le),\(l),\(rs),\(rw),\(re),\(r)")
                }
                   
            }
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
    options.landmarkMode = .none
    options.contourMode = .all
    options.classificationMode = .none
    options.performanceMode = .fast
    let faceDetector = FaceDetector.faceDetector(options: options)
    var faces: [Face]
    do {
      faces = try faceDetector.results(in: image)
    } catch let error {
      print("Failed to detect faces with error: \(error.localizedDescription).")
      self.updatePreviewOverlayViewWithLastFrame()
      return
    }
//    self.updatePreviewOverlayViewWithLastFrame()
    guard !faces.isEmpty else {
//      print("On-Device face detector returned no results.")
      return
    }
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      for face in faces {
//        let normalizedRect = CGRect(
//          x: face.frame.origin.x / width,
//          y: face.frame.origin.y / height,
//          width: face.frame.size.width / width,
//          height: face.frame.size.height / height
//        )
//        let standardizedRect = strongSelf.previewLayer.layerRectConverted(
//          fromMetadataOutputRect: normalizedRect
//        ).standardized
//        UIUtilities.addRectangle(
//          standardizedRect,
//          to: strongSelf.annotationOverlayView,
//          color: .clear
//        )
        strongSelf.addContours(for: face, width: width, height: height)
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
//        print("Pose detector returned no results.")
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
          let leftShoulderPosition = strongSelf.normalizedPoint(fromVisionPoint: pose.landmark(ofType: .leftShoulder).position, width: width, height: height)
          let leftElbowPosition = strongSelf.normalizedPoint(fromVisionPoint: pose.landmark(ofType: .leftElbow).position, width: width, height: height)
          let leftWristPosition = strongSelf.normalizedPoint(fromVisionPoint: pose.landmark(ofType: .leftWrist).position, width: width, height: height)
            
          let rightShoulderPosition = strongSelf.normalizedPoint(fromVisionPoint: pose.landmark(ofType: .rightShoulder).position, width: width, height: height)
          let rightElbowPosition = strongSelf.normalizedPoint(fromVisionPoint: pose.landmark(ofType: .rightElbow).position, width: width, height: height)
          let rightWristPosition = strongSelf.normalizedPoint(fromVisionPoint: pose.landmark(ofType: .rightWrist).position, width: width, height: height)
            if isRecording {
                data[curTS]?["leftShoulderCoor"] = "\(leftShoulderPosition.x):\(leftShoulderPosition.y)"
                data[curTS]?["leftWristCoor"] = "\(leftWristPosition.x):\(leftWristPosition.y)"
                data[curTS]?["leftElbowCoor"] = "\(leftElbowPosition.x):\(leftElbowPosition.y)"
                data[curTS]?["rightShoulderCoor"] = "\(rightShoulderPosition.x):\(rightShoulderPosition.y)"
                data[curTS]?["rightWristCoor"] = "\(rightWristPosition.x):\(rightWristPosition.y)"
                data[curTS]?["rightElbowCoor"] = "\(rightElbowPosition.x):\(rightElbowPosition.y)"
            }
          if leftElbowPosition.y < leftShoulderPosition.y || leftWristPosition.y < leftShoulderPosition.y {
              if isRecording {
                  data[curTS]?["leftHandMovement"] = 1
              }
              leftHandState.text = "Up"
          }else{
              if isRecording {
                  data[curTS]?["leftHandMovement"] = 0
              }
              leftHandState.text = "Down"
          }
          if rightElbowPosition.y < rightShoulderPosition.y || rightWristPosition.y < rightShoulderPosition.y {
              if isRecording {
                  data[curTS]?["rightHandMovement"] = 1
              }
              rightHandState.text = "Up"
          }else{
              if isRecording {
                  data[curTS]?["rightHandMovement"] = 0
              }
              rightHandState.text = "Down"
          }
            
//          let poseOverlayView = UIUtilities.createPoseOverlayView(
//            forPose: pose,
//            inViewWithBounds: strongSelf.annotationOverlayViewB.bounds,
//            lineWidth: Constant.lineWidth,
//            dotRadius: Constant.smallDotRadius,
//            positionTransformationClosure: { (position) -> CGPoint in
//              return strongSelf.normalizedPoint(
//                fromVisionPoint: position, width: width, height: height)
//            }
//          )
//          strongSelf.annotationOverlayViewB.addSubview(poseOverlayView)
        }
      }
    }
  }

  private func detectSegmentationMask(in image: VisionImage, sampleBuffer: CMSampleBuffer) {
    guard let segmenter = self.segmenter else {
      return
    }
    var mask: SegmentationMask
    do {
      mask = try segmenter.results(in: image)
    } catch let error {
      print("Failed to perform segmentation with error: \(error.localizedDescription).")
      self.updatePreviewOverlayViewWithLastFrame()
      return
    }
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.removeDetectionAnnotations()

      guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        print("Failed to get image buffer from sample buffer.")
        return
      }

      UIUtilities.applySegmentationMask(
        mask: mask, to: imageBuffer,
        backgroundColor: .white,
        foregroundColor: nil)
      strongSelf.updatePreviewOverlayViewWithImageBuffer(imageBuffer)
    }

  }

  private func detectObjectsOnDevice(
    in image: VisionImage,
    width: CGFloat,
    height: CGFloat,
    options: CommonObjectDetectorOptions
  ) {
    let detector = ObjectDetector.objectDetector(options: options)
    var objects: [Object]
    do {
      objects = try detector.results(in: image)
    } catch let error {
      print("Failed to detect objects with error: \(error.localizedDescription).")
      self.updatePreviewOverlayViewWithLastFrame()
      return
    }
    self.updatePreviewOverlayViewWithLastFrame()
    guard !objects.isEmpty else {
      print("On-Device object detector returned no results.")
      return
    }

    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      for object in objects {
        let normalizedRect = CGRect(
          x: object.frame.origin.x / width,
          y: object.frame.origin.y / height,
          width: object.frame.size.width / width,
          height: object.frame.size.height / height
        )
        let standardizedRect = strongSelf.previewLayer.layerRectConverted(
          fromMetadataOutputRect: normalizedRect
        ).standardized
        UIUtilities.addRectangle(
          standardizedRect,
          to: strongSelf.annotationOverlayView,
          color: UIColor.green
        )
        let label = UILabel(frame: standardizedRect)
        var description = ""
        if let trackingID = object.trackingID {
          description += "Object ID: " + trackingID.stringValue + "\n"
        }
        description += object.labels.enumerated().map { (index, label) in
          "Label \(index): \(label.text), \(label.confidence), \(label.index)"
        }.joined(separator: "\n")

        label.text = description
        label.numberOfLines = 0
        label.adjustsFontSizeToFitWidth = true
        strongSelf.rotate(label, orientation: image.orientation)
        strongSelf.annotationOverlayView.addSubview(label)
      }
    }
  }

  // MARK: - Private

  private func setUpCaptureSessionOutput() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.captureSession.beginConfiguration()
      // When performing latency tests to determine ideal capture settings,
      // run the app in 'release' mode to get accurate performance metrics
      strongSelf.captureSession.sessionPreset = AVCaptureSession.Preset.medium

      let output = AVCaptureVideoDataOutput()
      output.videoSettings = [
        (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA
      ]
      output.alwaysDiscardsLateVideoFrames = true
      let outputQueue = DispatchQueue(label: Constant.videoDataOutputQueueLabel)
      output.setSampleBufferDelegate(strongSelf, queue: outputQueue)
      guard strongSelf.captureSession.canAddOutput(output) else {
        print("Failed to add capture session output.")
        return
      }
      strongSelf.captureSession.addOutput(output)
      strongSelf.captureSession.commitConfiguration()
    }
  }

  private func setUpCaptureSessionInput() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      let cameraPosition: AVCaptureDevice.Position = strongSelf.isUsingFrontCamera ? .front : .back
      guard let device = strongSelf.captureDevice(forPosition: cameraPosition) else {
        print("Failed to get capture device for camera position: \(cameraPosition)")
        return
      }
      do {
        strongSelf.captureSession.beginConfiguration()
        let currentInputs = strongSelf.captureSession.inputs
        for input in currentInputs {
          strongSelf.captureSession.removeInput(input)
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard strongSelf.captureSession.canAddInput(input) else {
          print("Failed to add capture session input.")
          return
        }
        strongSelf.captureSession.addInput(input)
        strongSelf.captureSession.commitConfiguration()
      } catch {
        print("Failed to create capture device input: \(error.localizedDescription)")
      }
    }
  }

  private func startSession() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.captureSession.startRunning()
    }
  }

  private func stopSession() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.captureSession.stopRunning()
    }
  }

  private func setUpPreviewOverlayView() {
    cameraView.addSubview(previewOverlayView)
    NSLayoutConstraint.activate([
      previewOverlayView.centerXAnchor.constraint(equalTo: cameraView.centerXAnchor),
      previewOverlayView.centerYAnchor.constraint(equalTo: cameraView.centerYAnchor),
      previewOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
      previewOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),

    ])
  }

  private func setUpAnnotationOverlayView() {
    cameraView.addSubview(annotationOverlayView)
    NSLayoutConstraint.activate([
      annotationOverlayView.topAnchor.constraint(equalTo: cameraView.topAnchor),
      annotationOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
      annotationOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
      annotationOverlayView.bottomAnchor.constraint(equalTo: cameraView.bottomAnchor),
    ])
      cameraView.addSubview(annotationOverlayViewB)
      NSLayoutConstraint.activate([
        annotationOverlayViewB.topAnchor.constraint(equalTo: cameraView.topAnchor),
        annotationOverlayViewB.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
        annotationOverlayViewB.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
        annotationOverlayViewB.bottomAnchor.constraint(equalTo: cameraView.bottomAnchor),
      ])
  }

  private func captureDevice(forPosition position: AVCaptureDevice.Position) -> AVCaptureDevice? {
    if #available(iOS 10.0, *) {
      let discoverySession = AVCaptureDevice.DiscoverySession(
        deviceTypes: [.builtInWideAngleCamera],
        mediaType: .video,
        position: .unspecified
      )
      return discoverySession.devices.first { $0.position == position }
    }
    return nil
  }

  private func presentDetectorsAlertController() {
    let alertController = UIAlertController(
      title: Constant.alertControllerTitle,
      message: Constant.alertControllerMessage,
      preferredStyle: .alert
    )
    weak var weakSelf = self
    detectors.forEach { detectorType in
      let action = UIAlertAction(title: detectorType.rawValue, style: .default) {
        [unowned self] (action) in
        guard let value = action.title else { return }
        guard let detector = Detector(rawValue: value) else { return }
        guard let strongSelf = weakSelf else {
          print("Self is nil!")
          return
        }
        strongSelf.currentDetector = detector
        strongSelf.removeDetectionAnnotations()
      }
      if detectorType.rawValue == self.currentDetector.rawValue { action.isEnabled = false }
      alertController.addAction(action)
    }
    alertController.addAction(UIAlertAction(title: Constant.cancelActionTitleText, style: .cancel))
    present(alertController, animated: true)
  }

  private func removeDetectionAnnotations() {
    for annotationView in annotationOverlayView.subviews {
      annotationView.removeFromSuperview()
    }
      for annotationView in annotationOverlayViewB.subviews {
        annotationView.removeFromSuperview()
      }
  }

  private func updatePreviewOverlayViewWithLastFrame() {
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }

      guard let lastFrame = lastFrame,
        let imageBuffer = CMSampleBufferGetImageBuffer(lastFrame)
      else {
        return
      }
      strongSelf.updatePreviewOverlayViewWithImageBuffer(imageBuffer)
      strongSelf.removeDetectionAnnotations()
    }
  }

  private func updatePreviewOverlayViewWithImageBuffer(_ imageBuffer: CVImageBuffer?) {
    guard let imageBuffer = imageBuffer else {
      return
    }
    let orientation: UIImage.Orientation = isUsingFrontCamera ? .leftMirrored : .right
    let image = UIUtilities.createUIImage(from: imageBuffer, orientation: orientation)
    previewOverlayView.image = image
  }

  private func convertedPoints(
    from points: [NSValue]?,
    width: CGFloat,
    height: CGFloat
  ) -> [NSValue]? {
    return points?.map {
      let cgPointValue = $0.cgPointValue
      let normalizedPoint = CGPoint(x: cgPointValue.x / width, y: cgPointValue.y / height)
      let cgPoint = previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
      let value = NSValue(cgPoint: cgPoint)
      return value
    }
  }

  private func normalizedPoint(
    fromVisionPoint point: VisionPoint,
    width: CGFloat,
    height: CGFloat
  ) -> CGPoint {
    let cgPoint = CGPoint(x: point.x, y: point.y)
    var normalizedPoint = CGPoint(x: cgPoint.x / width, y: cgPoint.y / height)
    normalizedPoint = previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
    return normalizedPoint
  }

  private func addContours(for face: Face, width: CGFloat, height: CGFloat) {
    // Face
//    if let faceContour = face.contour(ofType: .face) {
//      for point in faceContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.blue,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }

    // Eyebrows
//    if let topLeftEyebrowContour = face.contour(ofType: .leftEyebrowTop) {
//      for point in topLeftEyebrowContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.orange,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
//    if let bottomLeftEyebrowContour = face.contour(ofType: .leftEyebrowBottom) {
//      for point in bottomLeftEyebrowContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.orange,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
//    if let topRightEyebrowContour = face.contour(ofType: .rightEyebrowTop) {
//      for point in topRightEyebrowContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.orange,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
//    if let bottomRightEyebrowContour = face.contour(ofType: .rightEyebrowBottom) {
//      for point in bottomRightEyebrowContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.orange,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }

    // Eyes
//    if let leftEyeContour = face.contour(ofType: .leftEye) {
//      for point in leftEyeContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.cyan,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
//    if let rightEyeContour = face.contour(ofType: .rightEye) {
//      for point in rightEyeContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.cyan,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }

    // Lips
      var averageUpperLipX = CGFloat.zero
    var averageUpperLipY = CGFloat.zero
    var averageLowerLipX = CGFloat.zero
    var averageLowerLipY = CGFloat.zero
      
    var upperLipCount = CGFloat.zero
    var bottomLipCount = CGFloat.zero
      
//    if let topUpperLipContour = face.contour(ofType: .upperLipTop) {
//      for point in topUpperLipContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.red,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
    if let bottomUpperLipContour = face.contour(ofType: .upperLipBottom) {
      for point in bottomUpperLipContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        upperLipCount += 1
          // upperLipCount = upperLipCount + 1
        averageUpperLipX += cgPoint.x
        averageUpperLipY += cgPoint.y
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.red,
//          radius: Constant.smallDotRadius
//        )
      }
    }
    averageUpperLipX /= upperLipCount
    averageUpperLipY /= upperLipCount
    
    if let topLowerLipContour = face.contour(ofType: .lowerLipTop) {
      for point in topLowerLipContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        bottomLipCount += 1
        averageLowerLipX += cgPoint.x
        averageLowerLipY += cgPoint.y
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.red,
//          radius: Constant.smallDotRadius
//        )
      }
    }
    
    averageLowerLipX /= bottomLipCount
    averageLowerLipY /= bottomLipCount
      if isRecording {
          data[curTS]?["mouthOpenness"] = abs(averageLowerLipY - averageUpperLipY)
      }
      if abs(Int(averageLowerLipY) - Int(averageUpperLipY)) <= Int(mouthThreshold.value) {
          if isRecording {
              data[curTS]?["mouthMovement"] = 0
          }
          mouthState.text = "Closed"
      }else{
          if isRecording {
              data[curTS]?["mouthMovement"] = 1
          }
          mouthState.text = "Opened"
      }
      
//    if let bottomLowerLipContour = face.contour(ofType: .lowerLipBottom) {
//      for point in bottomLowerLipContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.red,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }

    // Nose
//    if let noseBridgeContour = face.contour(ofType: .noseBridge) {
//      for point in noseBridgeContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.yellow,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
//    if let noseBottomContour = face.contour(ofType: .noseBottom) {
//      for point in noseBottomContour.points {
//        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
//        UIUtilities.addCircle(
//          atPoint: cgPoint,
//          to: annotationOverlayView,
//          color: UIColor.yellow,
//          radius: Constant.smallDotRadius
//        )
//      }
//    }
  }

  /// Resets any detector instances which use a conventional lifecycle paradigm. This method is
  /// expected to be invoked on the AVCaptureOutput queue - the same queue on which detection is
  /// run.
  private func resetManagedLifecycleDetectors() {
      self.poseDetector = PoseDetector.poseDetector(options: AccuratePoseDetectorOptions())
      self.segmenter = Segmenter.segmenter(options: SelfieSegmenterOptions())
      
//    if activeDetector == self.lastDetector {
//      // Same row as before, no need to reset any detectors.
//      return
//    }
//    // Clear the old detector, if applicable.
//    switch self.lastDetector {
//    case .pose, .poseAccurate:
//      self.poseDetector = nil
//      break
//    case .segmentationSelfie:
//      self.segmenter = nil
//      break
//    default:
//      break
//    }
//    // Initialize the new detector, if applicable.
//    switch activeDetector {
//    case .pose, .poseAccurate:
//      // The `options.detectorMode` defaults to `.stream`
//      let options = activeDetector == .pose ? PoseDetectorOptions() : AccuratePoseDetectorOptions()
//      self.poseDetector = PoseDetector.poseDetector(options: options)
//      break
//    case .segmentationSelfie:
//      // The `options.segmenterMode` defaults to `.stream`
//      let options = SelfieSegmenterOptions()
//      self.segmenter = Segmenter.segmenter(options: options)
//      break
//    default:
//      break
//    }
//    self.lastDetector = activeDetector
  }

  private func rotate(_ view: UIView, orientation: UIImage.Orientation) {
    var degree: CGFloat = 0.0
    switch orientation {
    case .up, .upMirrored:
      degree = 90.0
    case .rightMirrored, .left:
      degree = 180.0
    case .down, .downMirrored:
      degree = 270.0
    case .leftMirrored, .right:
      degree = 0.0
    }
    view.transform = CGAffineTransform.init(rotationAngle: degree * 3.141592654 / 180)
  }
}

// MARK: AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {

  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      print("Failed to get image buffer from sample buffer.")
      return
    }
    // Evaluate `self.currentDetector` once to ensure consistency throughout this method since it
    // can be concurrently modified from the main thread.
//    let activeDetector = self.currentDetector
//      resetManagedLifecycleDetectors(activeDetector: .onDeviceFace)
//      resetManagedLifecycleDetectors(activeDetector: .poseAccurate)
//      resetManagedLifecycleDetectors(activeDetector: .segmentationSelfie)
//      resetManagedLifecycleDetectors()

    lastFrame = sampleBuffer
    let visionImage = VisionImage(buffer: sampleBuffer)
    let orientation = UIUtilities.imageOrientation(
      fromDevicePosition: isUsingFrontCamera ? .front : .back
    )
    visionImage.orientation = orientation

    guard let inputImage = MLImage(sampleBuffer: sampleBuffer) else {
      print("Failed to create MLImage from sample buffer.")
      return
    }
    inputImage.orientation = orientation

    let imageWidth = CGFloat(CVPixelBufferGetWidth(imageBuffer))
    let imageHeight = CGFloat(CVPixelBufferGetHeight(imageBuffer))
    var shouldEnableClassification = false
    var shouldEnableMultipleObjects = false
      if isRecording {
          curTS = Int(Date().timeIntervalSince1970 * 1000)
          data[curTS] = [
            "mouthOpenness": "",
            "mouthMovement": "",
            "leftShoulderCoor": "",
            "leftWristCoor": "",
            "leftElbowCoor": "",
            "leftHandMovement": "",
            "rightShoulderCoor": "",
            "rightWristCoor": "",
            "rightElbowCoor": "",
            "rightHandMovement": "",
          ]
      }
    detectFacesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
    detectPose(in: inputImage, width: imageWidth, height: imageHeight)
//    detectSegmentationMask(in: visionImage, sampleBuffer: sampleBuffer)
      
//    switch activeDetector {
//    case .onDeviceObjectProminentWithClassifier, .onDeviceObjectMultipleWithClassifier,
//      .onDeviceObjectCustomProminentWithClassifier, .onDeviceObjectCustomMultipleWithClassifier:
//      shouldEnableClassification = true
//    default:
//      break
//    }
//    switch activeDetector {
//    case .onDeviceObjectMultipleNoClassifier, .onDeviceObjectMultipleWithClassifier,
//      .onDeviceObjectCustomMultipleNoClassifier, .onDeviceObjectCustomMultipleWithClassifier:
//      shouldEnableMultipleObjects = true
//    default:
//      break
//    }
//
//    switch activeDetector {
//    case .onDeviceBarcode:
//      scanBarcodesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
//    case .onDeviceFace:
//      detectFacesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
//    case .onDeviceText, .onDeviceTextChinese, .onDeviceTextDevanagari, .onDeviceTextJapanese,
//      .onDeviceTextKorean:
//      recognizeTextOnDevice(
//        in: visionImage, width: imageWidth, height: imageHeight, detectorType: activeDetector)
//    case .onDeviceImageLabel:
//      detectLabels(
//        in: visionImage, width: imageWidth, height: imageHeight, shouldUseCustomModel: false)
//    case .onDeviceImageLabelsCustom:
//      detectLabels(
//        in: visionImage, width: imageWidth, height: imageHeight, shouldUseCustomModel: true)
//    case .onDeviceObjectProminentNoClassifier, .onDeviceObjectProminentWithClassifier,
//      .onDeviceObjectMultipleNoClassifier, .onDeviceObjectMultipleWithClassifier:
//      // The `options.detectorMode` defaults to `.stream`
//      let options = ObjectDetectorOptions()
//      options.shouldEnableClassification = shouldEnableClassification
//      options.shouldEnableMultipleObjects = shouldEnableMultipleObjects
//      detectObjectsOnDevice(
//        in: visionImage,
//        width: imageWidth,
//        height: imageHeight,
//        options: options)
//    case .onDeviceObjectCustomProminentNoClassifier, .onDeviceObjectCustomProminentWithClassifier,
//      .onDeviceObjectCustomMultipleNoClassifier, .onDeviceObjectCustomMultipleWithClassifier:
//      guard
//        let localModelFilePath = Bundle.main.path(
//          forResource: Constant.localModelFile.name,
//          ofType: Constant.localModelFile.type
//        )
//      else {
//        print("Failed to find custom local model file.")
//        return
//      }
//      let localModel = LocalModel(path: localModelFilePath)
//      // The `options.detectorMode` defaults to `.stream`
//      let options = CustomObjectDetectorOptions(localModel: localModel)
//      options.shouldEnableClassification = shouldEnableClassification
//      options.shouldEnableMultipleObjects = shouldEnableMultipleObjects
//      detectObjectsOnDevice(
//        in: visionImage,
//        width: imageWidth,
//        height: imageHeight,
//        options: options)
//
//    case .pose, .poseAccurate:
//      detectPose(in: inputImage, width: imageWidth, height: imageHeight)
//    case .segmentationSelfie:
//      detectSegmentationMask(in: visionImage, sampleBuffer: sampleBuffer)
//    }
  }
}

// MARK: - Constants

public enum Detector: String {
  case onDeviceBarcode = "Barcode Scanning"
  case onDeviceFace = "Face Detection"
  case onDeviceText = "Text Recognition"
  case onDeviceTextChinese = "Text Recognition Chinese"
  case onDeviceTextDevanagari = "Text Recognition Devanagari"
  case onDeviceTextJapanese = "Text Recognition Japanese"
  case onDeviceTextKorean = "Text Recognition Korean"
  case onDeviceImageLabel = "Image Labeling"
  case onDeviceImageLabelsCustom = "Image Labeling Custom"
  case onDeviceObjectProminentNoClassifier = "ODT, single, no labeling"
  case onDeviceObjectProminentWithClassifier = "ODT, single, labeling"
  case onDeviceObjectMultipleNoClassifier = "ODT, multiple, no labeling"
  case onDeviceObjectMultipleWithClassifier = "ODT, multiple, labeling"
  case onDeviceObjectCustomProminentNoClassifier = "ODT, custom, single, no labeling"
  case onDeviceObjectCustomProminentWithClassifier = "ODT, custom, single, labeling"
  case onDeviceObjectCustomMultipleNoClassifier = "ODT, custom, multiple, no labeling"
  case onDeviceObjectCustomMultipleWithClassifier = "ODT, custom, multiple, labeling"
  case pose = "Pose Detection"
  case poseAccurate = "Pose Detection, accurate"
  case segmentationSelfie = "Selfie Segmentation"
}

private enum Constant {
  static let alertControllerTitle = "Vision Detectors"
  static let alertControllerMessage = "Select a detector"
  static let cancelActionTitleText = "Cancel"
  static let videoDataOutputQueueLabel = "com.google.mlkit.visiondetector.VideoDataOutputQueue"
  static let sessionQueueLabel = "com.google.mlkit.visiondetector.SessionQueue"
  static let noResultsMessage = "No Results"
  static let localModelFile = (name: "bird", type: "tflite")
  static let labelConfidenceThreshold = 0.75
  static let smallDotRadius: CGFloat = 4.0
  static let lineWidth: CGFloat = 3.0
  static let originalScale: CGFloat = 1.0
  static let padding: CGFloat = 10.0
  static let resultsLabelHeight: CGFloat = 200.0
  static let resultsLabelLines = 5
  static let imageLabelResultFrameX = 0.4
  static let imageLabelResultFrameY = 0.1
  static let imageLabelResultFrameWidth = 0.5
  static let imageLabelResultFrameHeight = 0.8
  static let segmentationMaskAlpha: CGFloat = 0.5
}
