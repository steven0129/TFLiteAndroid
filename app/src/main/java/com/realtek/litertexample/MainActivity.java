package com.realtek.litertexample;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Comparator;

public class MainActivity extends AppCompatActivity {

    private static String TAG="TEST";
    private Button button = null;
    private Interpreter tfliteInterpreter = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        this.button = findViewById(R.id.button);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        try {
            this.tfliteInterpreter = new Interpreter(loadModelFile(this, "MobileNetV2.tflite"));
        } catch(Exception ex) {
            Log.d(TAG, "Prepare Interpreter failed");
        }

        this.button.setOnClickListener(v -> {
            float[] result = runInference();
            String category = getTopCategory(result);
            List<String> topCategories = getTopCategories(result);
            Log.d(TAG, "Predicted Category: " + category);
            for (String c : topCategories) {
                Log.d(TAG, "Predicted Category: " + c);
            }
        });
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[] runInference() {
        // 從資源中載入圖片
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.cat_resized);

        // 初始化 TensorImage
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);

        // 使用 ImageProcessor 進行影像處理
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(127.5f, 127.5f))
                .build();
        tensorImage = imageProcessor.process(tensorImage);

        // 準備輸出緩衝區
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.FLOAT32);

        // 執行推論
        tfliteInterpreter.run(tensorImage.getBuffer(), outputBuffer.getBuffer());

        // 獲取結果
        return outputBuffer.getFloatArray();
    }

    private List<String> loadLabelsFromJson(Context context) {
        List<String> labels = new ArrayList<>();
        try {
            // 讀取 JSON 檔案
            InputStream inputStream = context.getAssets().open("labels.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder jsonString = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonString.append(line);
            }
            reader.close();

            // 解析 JSON 陣列
            JSONArray jsonArray = new JSONArray(jsonString.toString());
            for (int i = 0; i < jsonArray.length(); i++) {
                labels.add(jsonArray.getString(i));
            }
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Error reading labels JSON", e);
        }
        return labels;
    }

    private String getTopCategory(float[] result) {
        // 找到最大值的索引
        int maxIndex = 0;
        float maxScore = result[0];
        for (int i = 1; i < result.length; i++) {
            if (result[i] > maxScore) {
                maxScore = result[i];
                maxIndex = i;
            }
        }

        // 讀取 JSON 標籤
        List<String> labels = loadLabelsFromJson(getApplicationContext());
        if (maxIndex < labels.size()) {
            return labels.get(maxIndex);
        } else {
            return "Unknown";
        }
    }

    private List<String> getTopCategories(float[] result) {
        // 建立索引陣列
        Integer[] indices = new Integer[result.length];
        for (int i = 0; i < result.length; i++) {
            indices[i] = i;
        }

        // 根據機率進行降序排序
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                return Float.compare(result[i2], result[i1]);
            }
        });

        // 取得 JSON 標籤
        List<String> labels = loadLabelsFromJson(getApplicationContext());

        // 取出前5名的結果
        List<String> topCategories = new ArrayList<>();
        int numResults = Math.min(5, result.length);
        for (int i = 0; i < numResults; i++) {
            int idx = indices[i];
            float probability = result[idx];
            String category = (idx < labels.size()) ? labels.get(idx) : "Unknown";
            topCategories.add(category + ": " + probability);
        }
        return topCategories;
    }
}