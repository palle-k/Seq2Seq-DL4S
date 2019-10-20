// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NMTSwift",
    platforms: [
        .macOS(.v10_15), .iOS(.v13), .tvOS(.v13)
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/palle-k/DL4S.git", .branch("develop")),
        .package(url: "https://github.com/kylef/Commander.git", from: "0.8.0"),
        .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
        .package(url: "https://github.com/IBM-Swift/Kitura", from: "2.7.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "NMTSwift",
            dependencies: ["DL4S", "Commander", "PythonKit", "Kitura"]
        ),
        .testTarget(
            name: "NMTSwiftTests",
            dependencies: ["NMTSwift", "DL4S", "Commander"]
        ),
    ]
)
