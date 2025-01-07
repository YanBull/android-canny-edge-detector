package com.example.CannyFilter2

import android.Manifest
import android.content.ContentValues.TAG
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.CannyFilter2.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {

        supportActionBar?.hide()

        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded")

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        imageView = findViewById(R.id.imageView)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }
    }

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA



            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1920, 1080))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), ImageAnalysis.Analyzer { imageProxy ->
                processImage(imageProxy)
            })

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector,
                    imageAnalysis)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    private fun processImage(imageProxy: ImageProxy) {
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val bitmap = imageProxy.toBitmap() ?: return
        val rotatedBitmap = rotateBitmap(bitmap, rotationDegrees)

        val edgeBitmap = applyCannyEdgeDetection(rotatedBitmap)

        runOnUiThread {
            imageView.setImageBitmap(edgeBitmap)
        }

        imageProxy.close()
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val byteArray = out.toByteArray()
        return BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)
    }

    private fun applyCannyEdgeDetection(bitmap: Bitmap): Bitmap {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 100.0, 200.0)

        // Apply thresholding to filter out noise and retain consistent lines
        val thresholded = Mat()
        Imgproc.threshold(edges, thresholded, 128.0, 255.0, Imgproc.THRESH_BINARY)

        val edgeBitmap = Bitmap.createBitmap(thresholded.cols(), thresholded.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(thresholded, edgeBitmap)

        src.release()
        gray.release()
        edges.release()
        thresholded.release()

        return edgeBitmap
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)

    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
}