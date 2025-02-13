package com.realtek.litertexample;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
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
    private ImageView imageView = null;
    private TextView resultTextView = null;
    private Spinner delegateSpinner = null;
    private Interpreter tfliteInterpreter = null;
    private NnApiDelegate nnApiDelegate = null;
    private GpuDelegate gpuDelegate = null;
    private Bitmap bitmap = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        this.button = findViewById(R.id.button);
        this.imageView = findViewById(R.id.image_view);
        this.resultTextView = findViewById(R.id.result_text_view);
        this.delegateSpinner = findViewById(R.id.delegate_spinner);
        this.bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.cat_resized);
        this.imageView.setImageBitmap(bitmap);

        NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
        nnApiOptions.setAllowFp16(true).setUseNnapiCpu(true);
        this.nnApiDelegate = new NnApiDelegate(nnApiOptions);
        this.gpuDelegate = new GpuDelegate();

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        try {
            Interpreter.Options options = new Interpreter.Options();
            options.addDelegate(this.nnApiDelegate);
            this.tfliteInterpreter = new Interpreter(loadModelFile(this, "MobileNetV2.tflite"), options);
        } catch(Exception ex) {
            Log.d(TAG, "Prepare Interpreter failed");
        }

        this.button.setOnClickListener(v -> {
            float[] result = runInference(this.bitmap);
            String category = getTopCategory(result);
            List<String> topCategories = getTopCategories(result);
            this.resultTextView.setText("Predicted Category: " + category + ", inference with " + this.delegateSpinner.getSelectedItem().toString());

            Log.d(TAG, "Predicted Category: " + category);
            for (String c : topCategories) {
                Log.d(TAG, "Predicted Category: " + c);
            }
        });

        initSpinner();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.tfliteInterpreter != null) {
            this.tfliteInterpreter.close();
            this.tfliteInterpreter = null;
        }
        if (this.nnApiDelegate != null) {
            this.nnApiDelegate.close();
            this.nnApiDelegate = null;
        }
        if (this.gpuDelegate != null) {
            this.gpuDelegate.close();
            this.gpuDelegate = null;
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[] runInference(Bitmap bitmap) {
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(127.5f, 127.5f))
                .build();
        tensorImage = imageProcessor.process(tensorImage);
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.FLOAT32);
        tfliteInterpreter.run(tensorImage.getBuffer(), outputBuffer.getBuffer());
        return outputBuffer.getFloatArray();
    }

    private List<String> loadLabelsFromJson(Context context) {
        List<String> labels = new ArrayList<>();
        try {
            InputStream inputStream = context.getAssets().open("labels.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder jsonString = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonString.append(line);
            }
            reader.close();

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
        int maxIndex = 0;
        float maxScore = result[0];
        for (int i = 1; i < result.length; i++) {
            if (result[i] > maxScore) {
                maxScore = result[i];
                maxIndex = i;
            }
        }

        List<String> labels = loadLabelsFromJson(getApplicationContext());
        if (maxIndex < labels.size()) {
            return labels.get(maxIndex);
        } else {
            return "Unknown";
        }
    }

    private List<String> getTopCategories(float[] result) {
        Integer[] indices = new Integer[result.length];
        for (int i = 0; i < result.length; i++) {
            indices[i] = i;
        }

        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                return Float.compare(result[i2], result[i1]);
            }
        });

        List<String> labels = loadLabelsFromJson(getApplicationContext());
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

    private void initSpinner() {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"NNAPI", "GPU"});
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        delegateSpinner.setAdapter(adapter);

        delegateSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Interpreter.Options options = new Interpreter.Options();

                switch(position) {
                    case 0:
                        options.addDelegate(MainActivity.this.nnApiDelegate);
                        break;
                    case 1:
                        options.addDelegate(MainActivity.this.gpuDelegate);
                        break;
                }

                try {
                    MainActivity.this.tfliteInterpreter = new Interpreter(loadModelFile(MainActivity.this, "MobileNetV2.tflite"), options);
                } catch(IOException ex) {
                    Log.d(TAG, "Change interpreter failed");
                }

                MainActivity.this.resultTextView.setText("");
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }
}