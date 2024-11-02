using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictiveAIExample;

class Program
{
    static void Main()
    {
        // Paso 1: Crear un contexto de ML
        MLContext mlContext = new MLContext();

        // Paso 2: Cargar los datos de entrenamiento
        string dataPath = @"C:/Pruebas/data.csv";
        IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, separatorChar: ',', hasHeader: true);

        // Paso 3: Definir el pipeline de entrenamiento
        var pipeline = mlContext.Transforms.NormalizeMinMax("Size")
        .Append(mlContext.Transforms.Concatenate("Features", "Size"))
        .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 2000));

        // Paso 4: Entrenar el modelo
        var model = pipeline.Fit(dataView);

        var testMetrics = mlContext.Regression.Evaluate(model.Transform(dataView), labelColumnName: "Price");

        Console.WriteLine($"R^2: {testMetrics.RSquared:0.##}");
        Console.WriteLine($"MAE: {testMetrics.MeanAbsoluteError:0.##}");
        Console.WriteLine($"RMSE: {testMetrics.RootMeanSquaredError:0.##}");

        Console.WriteLine("");
        Console.WriteLine("");


        Console.WriteLine("Indiquenos cuantos metros tiene la casa que busca");

        // Paso 5: Crear una instancia de HouseData para la predicción
        var houseToPredict = new HouseData() { Size = float.Parse(Console.ReadLine().ToString()) };
        var prediction = Predict(mlContext, model, houseToPredict);

        // Mostrar resultado de la predicción
        Console.WriteLine($"Predicción para una casa de {houseToPredict.Size} pies cuadrados: ${prediction.Price:0.00}");
        Console.WriteLine("¿Este precio es adecuado? (Y/N)");
        bool validator = (Console.ReadLine().ToUpper().Trim() == "Y") ? true : false;
        
        if(validator)
        {
            mlContext.Model.Save(model, dataView.Schema, $"modeloPredictivo{DateTime.Now.ToString("ddMMyyyyhhmmss")}.zip");
            Console.WriteLine("Modelo guardado");
        }
        Console.WriteLine("Gracias por confiar en nosotros");
    }

    public static HousePricePrediction Predict(MLContext mlContext, ITransformer model, HouseData houseData)
    {
        // Crear un predictor
        var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);

        // Realizar la predicción
        var prediction = predictionEngine.Predict(houseData);

        return prediction;
    }
}
