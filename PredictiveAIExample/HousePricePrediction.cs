using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictiveAIExample
{
    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
