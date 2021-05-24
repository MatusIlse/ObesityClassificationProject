{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Project_IM.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyP3tFr2k6Dbsf9KhXbiS8dh",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MatusIlse/ObesityClassificationProject/blob/main/Project_IM.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lHcgsvNQOJCk",
        "outputId": "fdfa2c16-3578-482c-c6a0-6ea0e14bf4a4"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from google.colab import drive\n",
        "\n",
        "drive.mount(\"/content/gdrive\")  \n",
        "!pwd"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/gdrive\n",
            "/content\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hqtYo9amPzu8",
        "outputId": "bdb846d8-3969-4fa6-be3c-32503092e10c"
      },
      "source": [
        "%cd \"/content/gdrive/My Drive/8VO SEMESTRE/Sistemas inteligentes/PROJECT\"\n",
        "!ls"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/gdrive/My Drive/8VO SEMESTRE/Sistemas inteligentes/PROJECT\n",
            "'Copia de Project_IM.ipynb'\n",
            "'ObesityDataSet_raw_and_data_sinthetic (2).zip'\n",
            " ObesityDataSet_raw_and_data_sinthetic.arff\n",
            " ObesityDataSet_raw_and_data_sinthetic.csv\n",
            " Project_IM.ipynb\n",
            " Untitled0.ipynb\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nfQ1gxOsQeIO",
        "outputId": "13873a6d-bf3f-4bba-8c0c-e28f5ef11f9e"
      },
      "source": [
        "df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')\n",
        "print(df.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   Gender   Age  Height  ...        CALC                 MTRANS           NObeyesdad\n",
            "0  Female  21.0    1.62  ...          no  Public_Transportation        Normal_Weight\n",
            "1  Female  21.0    1.52  ...   Sometimes  Public_Transportation        Normal_Weight\n",
            "2    Male  23.0    1.80  ...  Frequently  Public_Transportation        Normal_Weight\n",
            "3    Male  27.0    1.80  ...  Frequently                Walking   Overweight_Level_I\n",
            "4    Male  22.0    1.78  ...   Sometimes  Public_Transportation  Overweight_Level_II\n",
            "\n",
            "[5 rows x 17 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 417
        },
        "id": "jLlzE7wTxLXO",
        "outputId": "41b3ae15-7288-4b2f-aab3-dc14fd870da1"
      },
      "source": [
        "df['NObeyesdad'].value_counts().plot.bar()\n",
        "plt.xlabel(\"Class\")\n",
        "plt.ylabel(\"People count\")\n",
        "plt.title(\"\"\"Levels of obesity\n",
        "Original data frame\"\"\")\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Levels of obesity\\nOriginal data frame')"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 405
        },
        "id": "WhVmNXjlveVN",
        "outputId": "68030dd0-b0fc-4b3c-ac27-00252b4c6223"
      },
      "source": [
        "plt.figure(figsize=(20,6))\n",
        "\n",
        "plt.subplot(131)\n",
        "plt.scatter(df['Age'],df['Weight'],  color='blue')\n",
        "plt.xlabel(\"Age\")\n",
        "plt.ylabel(\"Weight\")\n",
        "\n",
        "plt.subplot(132)\n",
        "plt.scatter(df['Height'],df['Weight'],  color='Orange')\n",
        "plt.xlabel(\"Height\")\n",
        "plt.ylabel(\"Weight\")\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'Weight')"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1440x432 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZVv4mynuTZYH",
        "outputId": "6d0bfb99-1318-40c3-ee63-08e811d45c30"
      },
      "source": [
        "df_y = df[['NObeyesdad']]\n",
        "print(df_y.head())\n",
        "\n",
        "df_bool=df[['Gender','family_history_with_overweight','FAVC','SMOKE','SCC']]\n",
        "print(df_bool.head())\n",
        "\n",
        "df_xstr=df[['CAEC','CALC','MTRANS']]\n",
        "print(df_xstr.head())\n",
        "\n",
        "df_xnum=df[['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']]\n",
        "print(df_xnum.head())\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "            NObeyesdad\n",
            "0        Normal_Weight\n",
            "1        Normal_Weight\n",
            "2        Normal_Weight\n",
            "3   Overweight_Level_I\n",
            "4  Overweight_Level_II\n",
            "   Gender family_history_with_overweight FAVC SMOKE  SCC\n",
            "0  Female                            yes   no    no   no\n",
            "1  Female                            yes   no   yes  yes\n",
            "2    Male                            yes   no    no   no\n",
            "3    Male                             no   no    no   no\n",
            "4    Male                             no   no    no   no\n",
            "        CAEC        CALC                 MTRANS\n",
            "0  Sometimes          no  Public_Transportation\n",
            "1  Sometimes   Sometimes  Public_Transportation\n",
            "2  Sometimes  Frequently  Public_Transportation\n",
            "3  Sometimes  Frequently                Walking\n",
            "4  Sometimes   Sometimes  Public_Transportation\n",
            "    Age  Height  Weight  FCVC  NCP  CH2O  FAF  TUE\n",
            "0  21.0    1.62    64.0   2.0  3.0   2.0  0.0  1.0\n",
            "1  21.0    1.52    56.0   3.0  3.0   3.0  3.0  0.0\n",
            "2  23.0    1.80    77.0   2.0  3.0   2.0  2.0  1.0\n",
            "3  27.0    1.80    87.0   3.0  3.0   2.0  2.0  0.0\n",
            "4  22.0    1.78    89.8   2.0  1.0   2.0  0.0  0.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "52n0GB0xr319",
        "outputId": "d457421b-9798-4ed7-cfb6-73e7f50d9ffc"
      },
      "source": [
        "from sklearn import preprocessing\n",
        "print(df_xstr)\n",
        "le = preprocessing.LabelEncoder()\n",
        "df_xEn = df_xstr.apply(le.fit_transform)\n",
        "print(df_xEn.head())\n",
        "type(df_xEn)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "           CAEC        CALC                 MTRANS\n",
            "0     Sometimes          no  Public_Transportation\n",
            "1     Sometimes   Sometimes  Public_Transportation\n",
            "2     Sometimes  Frequently  Public_Transportation\n",
            "3     Sometimes  Frequently                Walking\n",
            "4     Sometimes   Sometimes  Public_Transportation\n",
            "...         ...         ...                    ...\n",
            "2106  Sometimes   Sometimes  Public_Transportation\n",
            "2107  Sometimes   Sometimes  Public_Transportation\n",
            "2108  Sometimes   Sometimes  Public_Transportation\n",
            "2109  Sometimes   Sometimes  Public_Transportation\n",
            "2110  Sometimes   Sometimes  Public_Transportation\n",
            "\n",
            "[2111 rows x 3 columns]\n",
            "   CAEC  CALC  MTRANS\n",
            "0     2     3       3\n",
            "1     2     2       3\n",
            "2     2     1       3\n",
            "3     2     1       4\n",
            "4     2     2       3\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "pandas.core.frame.DataFrame"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8hs-qAH06IuD",
        "outputId": "e53cffe6-43be-49bd-ac1e-edd8db2aef36"
      },
      "source": [
        "print(df_bool)\n",
        "enc = preprocessing.OneHotEncoder()\n",
        "enc.fit(df_bool)\n",
        "df_xOHE = enc.transform(df_bool).toarray()\n",
        "df_xOHE.shape\n",
        "df_xOHE\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "      Gender family_history_with_overweight FAVC SMOKE  SCC\n",
            "0     Female                            yes   no    no   no\n",
            "1     Female                            yes   no   yes  yes\n",
            "2       Male                            yes   no    no   no\n",
            "3       Male                             no   no    no   no\n",
            "4       Male                             no   no    no   no\n",
            "...      ...                            ...  ...   ...  ...\n",
            "2106  Female                            yes  yes    no   no\n",
            "2107  Female                            yes  yes    no   no\n",
            "2108  Female                            yes  yes    no   no\n",
            "2109  Female                            yes  yes    no   no\n",
            "2110  Female                            yes  yes    no   no\n",
            "\n",
            "[2111 rows x 5 columns]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1., 0., 0., ..., 0., 1., 0.],\n",
              "       [1., 0., 0., ..., 1., 0., 1.],\n",
              "       [0., 1., 0., ..., 0., 1., 0.],\n",
              "       ...,\n",
              "       [1., 0., 0., ..., 0., 1., 0.],\n",
              "       [1., 0., 0., ..., 0., 1., 0.],\n",
              "       [1., 0., 0., ..., 0., 1., 0.]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SJsS5PA0EBLR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "65f86c53-ffff-451e-b0a9-9702f92da9a2"
      },
      "source": [
        "#aqui tengo que unir xnum con xHOE para crear un solo arreglo df_x con todos los atributos y sin valores string\n",
        "array_x=np.concatenate((df_xnum,df_xEn,df_xOHE),axis=1)\n",
        "print(array_x)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[ 21.         1.62      64.       ...   0.         1.         0.      ]\n",
            " [ 21.         1.52      56.       ...   1.         0.         1.      ]\n",
            " [ 23.         1.8       77.       ...   0.         1.         0.      ]\n",
            " ...\n",
            " [ 22.524036   1.752206 133.689352 ...   0.         1.         0.      ]\n",
            " [ 24.361936   1.73945  133.346641 ...   0.         1.         0.      ]\n",
            " [ 23.664709   1.738836 133.472641 ...   0.         1.         0.      ]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7K7y6qMhVirI",
        "outputId": "34dbee8f-59f2-4773-d9cf-9759737e5fc2"
      },
      "source": [
        "df_x = pd.DataFrame(array_x)\n",
        "df_x.columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','CAEC','CALC','MTRANS','Female','Male','Overweight history_No','Overweight history_Yes','FAVC_No','FAVC_Yes','SMOKE_No','SMOKE_Yes','SCC_No','SCC_Yes']\n",
        "print(df_x)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "            Age    Height      Weight  ...  SMOKE_Yes  SCC_No  SCC_Yes\n",
            "0     21.000000  1.620000   64.000000  ...        0.0     1.0      0.0\n",
            "1     21.000000  1.520000   56.000000  ...        1.0     0.0      1.0\n",
            "2     23.000000  1.800000   77.000000  ...        0.0     1.0      0.0\n",
            "3     27.000000  1.800000   87.000000  ...        0.0     1.0      0.0\n",
            "4     22.000000  1.780000   89.800000  ...        0.0     1.0      0.0\n",
            "...         ...       ...         ...  ...        ...     ...      ...\n",
            "2106  20.976842  1.710730  131.408528  ...        0.0     1.0      0.0\n",
            "2107  21.982942  1.748584  133.742943  ...        0.0     1.0      0.0\n",
            "2108  22.524036  1.752206  133.689352  ...        0.0     1.0      0.0\n",
            "2109  24.361936  1.739450  133.346641  ...        0.0     1.0      0.0\n",
            "2110  23.664709  1.738836  133.472641  ...        0.0     1.0      0.0\n",
            "\n",
            "[2111 rows x 21 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Iu1cpShj_oon",
        "outputId": "af2552d4-43ee-4808-d656-a57855c39948"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "dfx_train,dfx_test,dfy_train,dfy_test=train_test_split(df_x,df_y,test_size=0.1)\n",
        "print(\"\"\"Train data set\n",
        "\"\"\")\n",
        "print(dfx_train.head())\n",
        "print(dfy_train.head())\n",
        "print(\"\"\"\n",
        "Test data set\n",
        "\"\"\")\n",
        "print(dfx_test.head())\n",
        "print(dfy_test.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train data set\n",
            "\n",
            "            Age    Height      Weight  ...  SMOKE_Yes  SCC_No  SCC_Yes\n",
            "1926  18.120739  1.807576  152.567671  ...        0.0     1.0      0.0\n",
            "60    21.000000  1.550000   49.000000  ...        0.0     1.0      0.0\n",
            "757   21.845025  1.613668   68.126955  ...        0.0     1.0      0.0\n",
            "1116  30.079371  1.810616   92.860254  ...        0.0     1.0      0.0\n",
            "123   24.000000  1.660000   67.000000  ...        0.0     1.0      0.0\n",
            "\n",
            "[5 rows x 21 columns]\n",
            "               NObeyesdad\n",
            "1926     Obesity_Type_III\n",
            "60          Normal_Weight\n",
            "757    Overweight_Level_I\n",
            "1116  Overweight_Level_II\n",
            "123         Normal_Weight\n",
            "\n",
            "Test data set\n",
            "\n",
            "            Age    Height      Weight  ...  SMOKE_Yes  SCC_No  SCC_Yes\n",
            "723   18.281092  1.700000   50.000000  ...        0.0     1.0      0.0\n",
            "347   17.000000  1.800000   97.000000  ...        0.0     1.0      0.0\n",
            "1135  21.959940  1.483284   62.894283  ...        0.0     1.0      0.0\n",
            "355   27.000000  1.550000   62.000000  ...        0.0     1.0      0.0\n",
            "1554  21.963787  1.849601  122.333425  ...        0.0     1.0      0.0\n",
            "\n",
            "[5 rows x 21 columns]\n",
            "               NObeyesdad\n",
            "723   Insufficient_Weight\n",
            "347   Overweight_Level_II\n",
            "1135  Overweight_Level_II\n",
            "355    Overweight_Level_I\n",
            "1554      Obesity_Type_II\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nmssKD1MDvdB",
        "outputId": "1c7243e4-5d19-4b6d-9d99-6d76b78452a1"
      },
      "source": [
        "#hyper parameter max depth\n",
        "tree_clf = DecisionTreeClassifier(max_depth = 10)\n",
        "tree_clf.fit(dfx_train,dfy_train)\n",
        "\n",
        "print(\"tree classifier configuration\")\n",
        "print (tree_clf)\n",
        "\n",
        "#print(\"\")\n",
        "#print(\"Prediction for test data\")\n",
        "Test_predict = pd.DataFrame(tree_clf.predict(dfx_test))\n",
        "#print(Test_predict)\n",
        "#print(\"\")\n",
        "#print(\"Real class for test data\")\n",
        "#print(dfy_test)\n",
        "\n",
        "print(\"Accuracy:\",accuracy_score(dfy_test, Test_predict))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "tree classifier configuration\n",
            "DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',\n",
            "                       max_depth=10, max_features=None, max_leaf_nodes=None,\n",
            "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
            "                       min_samples_leaf=1, min_samples_split=2,\n",
            "                       min_weight_fraction_leaf=0.0, presort='deprecated',\n",
            "                       random_state=None, splitter='best')\n",
            "Accuracy: 0.9481132075471698\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 710
        },
        "id": "JYMrJXHkh8Eb",
        "outputId": "bdadc6da-7e2c-4f04-cfcd-243f744e7bde"
      },
      "source": [
        "plt.figure(figsize=(15, 3))\n",
        "plt.subplot(131)\n",
        "Test_predict.value_counts().plot.bar(color='Orange')\n",
        "plt.xlabel(\"Class\")\n",
        "plt.ylabel(\"People count\")\n",
        "plt.title(\"\"\"Prediction results \n",
        "Count of people per class \"\"\")\n",
        "plt.subplot(132)\n",
        "dfy_test.value_counts().plot.bar()\n",
        "plt.xlabel(\"Class\")\n",
        "plt.ylabel(\"People count\")\n",
        "plt.title(\"\"\"Original Test Data \n",
        "Count of people per class \"\"\")\n",
        "\n",
        "Val1=Test_predict.value_counts()\n",
        "Val2=dfy_test.value_counts()\n",
        "print(\"\"\"\n",
        "Prediction Results\"\"\")\n",
        "print(Val1)\n",
        "print(\"\"\"\n",
        "Original Test Data\"\"\")\n",
        "print(Val2)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Prediction Results\n",
            "Obesity_Type_I         39\n",
            "Overweight_Level_I     35\n",
            "Insufficient_Weight    30\n",
            "Normal_Weight          29\n",
            "Obesity_Type_III       28\n",
            "Obesity_Type_II        26\n",
            "Overweight_Level_II    25\n",
            "dtype: int64\n",
            "\n",
            "Original Test Data\n",
            "NObeyesdad         \n",
            "Obesity_Type_I         39\n",
            "Overweight_Level_I     33\n",
            "Insufficient_Weight    31\n",
            "Normal_Weight          29\n",
            "Obesity_Type_III       28\n",
            "Overweight_Level_II    26\n",
            "Obesity_Type_II        26\n",
            "dtype: int64\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x216 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_xoQaQV_H762",
        "outputId": "63afcb91-3299-4ac3-baea-0b6a0d97a460"
      },
      "source": [
        "#Here we can feed it with a new instance \n",
        "Array_Ans=[None]*11\n",
        "Array_Ans[0] = int (input('Age:'))\n",
        "Array_Ans[1] = float (input('Height in meters:'))\n",
        "Array_Ans[2] = int (input('Weight in kg:'))\n",
        "\n",
        "print(\"Never=3,Sometimes=2,Alwas=1\")\n",
        "Array_Ans[3] = int (input('Do you usually eat vegetables in ypur meals?'))\n",
        "\n",
        "print(\"One to two=3,  Three=2  , More than three=1\")\n",
        "Array_Ans[4] = int (input('How many main meals do you have daily?'))\n",
        "\n",
        "print(\"Less than a liter=3,  One or two liters=2 , More than two liters=1\")\n",
        "Array_Ans[5] = int (input('How much water do you drink daily?'))\n",
        "\n",
        "print(\"0 to 1 day= 3,  1 or to days=2,  2 or 4 days=1,  More than 4: 0\")\n",
        "Array_Ans[6] = int (input('How often do you excercise?'))\n",
        "\n",
        "print(\"0 to 2 hours=2,  3 to 5 hours=1,  More than 5: 0\")\n",
        "Array_Ans[7] = int (input('Hours spent using electronic devices?'))\n",
        "\n",
        "print(\"no=3,Sometimes=2,Frequently=1,Always=0\")\n",
        "Array_Ans[8] = int(input('Do you eat between meals?'))\n",
        "\n",
        "print(\"Never=3,Sometimes=2,Frequently=1,Always=0\")\n",
        "Array_Ans[9] = int (input('How often do you dink alcohol?'))\n",
        "\n",
        "print(\"Automobile=0,Motorbike=1,Bike=2,Public_Transportation=3,Walking=4\")\n",
        "Array_Ans[10] = int (input('From the above, choose your transportation method:'))\n",
        "print(Array_Ans)\n",
        "\n",
        "Ans_str=[[None]*5]\n",
        "Ans_str[0][0] = str (input('Gender(female,male):'))\n",
        "Ans_str[0][1] = str (input('Are there any cases of obesity in your family?(yes/no):'))\n",
        "Ans_str[0][2] = str (input('Do you eat high caloric food frequently?(yes/no):'))\n",
        "Ans_str[0][3] = str (input('Do you smoke?(yes/no):'))\n",
        "Ans_str[0][4] = str (input('Do you monitor your calories?(yes/no):'))\n",
        "print(Ans_str)\n",
        "#Necesito hacer one hot encoding para en array de respuestas\n",
        "Array_OHE=[None]*10\n",
        "for i in range (5):\n",
        "  if i ==0:\n",
        "    if Ans_str[0][i]== 'female':\n",
        "      Array_OHE[0]=1\n",
        "      Array_OHE[1]=0\n",
        "    else:\n",
        "      Array_OHE[0]=0\n",
        "      Array_OHE[1]=1\n",
        "  else:\n",
        "    if  Ans_str[0][i]== 'no':\n",
        "      Array_OHE[i*2]=1\n",
        "      Array_OHE[(i*2)+1]=0\n",
        "    else:\n",
        "      Array_OHE[i*2]=0\n",
        "      Array_OHE[(i*2)+1]=1\n",
        "      \n",
        "print(Array_OHE)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Age:24\n",
            "Height in meters:1.6\n",
            "Weight in kg:50\n",
            "Never=3,Sometimes=2,Alwas=1\n",
            "Do you usually eat vegetables in ypur meals?3\n",
            "One to two=3,  Three=2  , More than three=1\n",
            "How many main meals do you have daily?1\n",
            "Less than a liter=3,  One or two liters=2 , More than two liters=1\n",
            "How much water do you drink daily?2\n",
            "0 to 1 day= 3,  1 or to days=2,  2 or 4 days=1,  More than 4: 0\n",
            "How often do you excercise?3\n",
            "0 to 2 hours=2,  3 to 5 hours=1,  More than 5: 0\n",
            "Hours spent using electronic devices?0\n",
            "no=3,Sometimes=2,Frequently=1,Always=0\n",
            "Do you eat between meals?0\n",
            "Never=3,Sometimes=2,Frequently=1,Always=0\n",
            "How often do you dink alcohol?0\n",
            "Automobile=0,Motorbike=1,Bike=2,Public_Transportation=3,Walking=4\n",
            "From the above, choose your transportation method:0\n",
            "[24, 1.6, 50, 3, 1, 2, 3, 0, 0, 0, 0]\n",
            "Gender(female,male):female\n",
            "Are there any cases of obesity in your family?(yes/no):yes\n",
            "Do you eat high caloric food frequently?(yes/no):yes\n",
            "Do you smoke?(yes/no):no\n",
            "Do you monitor your calories?(yes/no):no\n",
            "[['female', 'yes', 'yes', 'no', 'no']]\n",
            "[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PGFhcvs5YcUf",
        "outputId": "3cd929b7-58f3-48f3-dd87-38fe24abb10e"
      },
      "source": [
        "UserAns=np.concatenate((Array_Ans,Array_OHE))\n",
        "print(UserAns)\n",
        "#df_Ans=pd.DataFrame(Array_Ans,columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','CAEC','CALC','MTRANS','Female','Male','Overweight history_No','Overweight history_Yes','FAVC_No','FAVC_Yes','SMOKE_No','SMOKE_Yes','SCC_No','SCC_Yes'])\n",
        "#print(df_Ans)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[24.   1.6 50.   3.   1.   2.   3.   0.   0.   0.   0.   1.   0.   0.\n",
            "  1.   0.   1.   1.   0.   1.   0. ]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4Ou-9uydDv1E",
        "outputId": "57ba6b72-74c1-40d7-be3f-7cefdbe13a8b"
      },
      "source": [
        "#make user prediction\n",
        "\n",
        "print(\"Prediction for new user\")\n",
        "probs=pd.DataFrame(tree_clf.predict_proba([UserAns]))\n",
        "probs = tree_clf.predict_proba([UserAns])\n",
        "print(\"probability of class for user is\",probs)\n",
        "\n",
        "pred =  tree_clf.predict([UserAns])\n",
        "print(\"\"\"\n",
        "The user's weight was classified into the next class: \n",
        "\"\"\",pred)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Prediction for new user\n",
            "probability of class for user is [[0. 1. 0. 0. 0. 0. 0.]]\n",
            "\n",
            "The user's weight was classified into the next class: \n",
            " ['Normal_Weight']\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}