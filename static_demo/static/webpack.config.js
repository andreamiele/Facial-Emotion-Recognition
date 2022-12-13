
module.exports = {
  mode: "development",
  devtool: "source-map",
  entry: {
    home: "./src/js/home.js"
  },
  output: {
    path: __dirname + "/dist/js",
    filename: "[name].js"
  },
  module:{
    rules:[
      {
        test: /\.js$|jsx/,
        exclude: /node_modules/,
        use: 'babel-loader',
      },
    ],
  },
};