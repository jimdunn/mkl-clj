(defproject mkl-clj "0.1.0-SNAPSHOT"
  :description "Clojure library for using the Intel Math Kernel Library"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.bytedeco.javacpp-presets/mkl-platform "2018.1-1.4.1"]
                 [midje "1.9.1"]]
  :profiles {:dev {:dependencies [[midje "1.9.1"]]}})
