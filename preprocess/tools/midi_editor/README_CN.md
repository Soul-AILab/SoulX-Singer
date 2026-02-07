# 🎹 MIDI Editor - 网页端歌声 MIDI 编辑器

[English](README.md) | [简体中文](README_CN.md)

一个功能完整的网页端歌声 MIDI 文件编辑器。支持实时拖拽调整 MIDI 音符、歌词编辑、音频波形对齐，以及导入导出含歌词的 MIDI 文件。

![MIDI Editor](https://img.shields.io/badge/React-19.2-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue) ![Vite](https://img.shields.io/badge/Vite-7.2-purple)

## ✨ 功能特性

### 🎼 钢琴卷帘编辑

- **可视化音符编辑**：支持 C1-C8 全音域显示，直观的钢琴键布局
- **拖拽操作**：
  - 移动音符：拖拽音符块调整位置和音高
  - 调整音头：拖拽音符左边缘调整开始时间
  - 调整音尾：拖拽音符右边缘调整结束时间
- **快捷音高调整**：Command/Ctrl + 上/下键微调选中音符的音高
- **双击添加**：在钢琴卷帘空白处双击快速添加新音符
- **钢琴键试听**：点击左侧钢琴键可试听对应音高

### 🔍 缩放与导航

- **水平缩放**
- **垂直缩放**
- **动态精度**：缩放越大，音符调整的 snap 粒度越精细（最小 0.01 秒）
- **自动滚动**：播放时播放头自动保持可见

### 📝 歌词编辑

- **实时编辑**：右侧列表直接编辑每个音符的歌词
- **批量填充**：输入一段歌词，按字顺序自动填充到音符
- **从选中开始**：批量填充可从当前选中的音符开始
- **精确调整**：可直接编辑 PITCH（音高）、START（开始时间）、END（结束时间）
- **确认机制**：修改数值后按 Enter 或点击 ✓ 确认，避免误操作

### 🎵 音频对齐

- **波形显示**：导入音频后显示波形，与 MIDI 同步滚动
- **格式支持**：MP3、WAV、OGG、FLAC、M4A、AAC
- **同步播放**：音频与 MIDI 同步播放，可分别调整音量大小
- **点击定位**：点击波形或时间尺可快速定位播放位置

### ⚠️ 重叠检测

- **可视化标注**：时间重叠的音符显示为红色并闪烁
- **智能容差**：紧邻的音符（上一个结束 = 下一个开始）不视为重叠
- **一键修复**：点击消除重叠按钮自动修复所有重叠
- **导出提醒**：导出时如有重叠会弹出警告

### 📥 导入导出

- **MIDI 导入**：支持标准 MIDI 文件，自动解析歌词元数据
- **MIDI 导出**：导出包含歌词信息的 MIDI 文件
- **中文支持**：完整支持中文歌词的导入导出（UTF-8 编码）

### 🎨 界面特性

- **主题切换**：支持浅色/深色主题
- **响应式布局**：自适应窗口大小
- **SVG 网格**：跨浏览器兼容的网格渲染
- **状态提示**：实时显示操作状态和错误信息

## 🚀 快速开始

### 环境要求

- Node.js 18+
- npm 或 yarn

### 安装

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 在局域网启动
npm run dev -- --host 0.0.0.0
```

### 构建

```bash
# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

## 📖 使用指南

### 基本工作流

1. **导入 MIDI**：点击导入 MIDI 按钮选择 .mid 文件
2. **编辑音符**：在钢琴卷帘中拖拽调整音符位置和时长
3. **添加歌词**：在右侧列表中输入每个音符的歌词
4. **对齐音频**（可选）：导入参考音频进行对照编辑
5. **导出文件**：点击导出含歌词 MIDI 保存文件

### 快捷操作

| 操作 | 说明 |
|------|------|
| 双击钢琴卷帘 | 添加新音符 |
| 双击音符 | 修改歌词 |
| 拖拽音符 | 移动音符位置/音高 |
| 拖拽音符边缘 | 调整音符时长 |
| Backspace / Delete | 删除选中音符 |
| Enter | 确认数值修改 |
| Escape | 取消数值修改 |
| Ctrl(Command) + 滚轮 | 水平缩放 |
| Ctrl(Command) + Shift(Option) + 滚轮 | 垂直缩放 |

### 播放控制

| 按钮 | 功能 |
|------|------|
| ⏮ | 回到开头 |
| ⏪ 2s | 后退 2 秒 |
| ▶ / ⏸ | 播放 / 暂停 |
| 2s ⏩ | 前进 2 秒 |
| ⏭ | 跳到结尾 |
| 选定区域 | 播放选定区域 |

## 🛠 技术栈

- **前端框架**：React 19 + TypeScript
- **构建工具**：Vite 7
- **状态管理**：Zustand
- **音频引擎**：Tone.js
- **波形显示**：WaveSurfer.js
- **MIDI 解析**：@tonejs/midi
- **样式**：CSS（自定义变量主题）

## 📁 项目结构

```
.
├── eslint.config.js
├── index.html
├── package.json
├── postcss.config.js
├── README.md
├── README_CN.md
├── tailwind.config.js
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── public/
└── src/
    ├── App.css
    ├── App.tsx
    ├── constants.ts
    ├── index.css
    ├── main.tsx
    ├── types.ts
    ├── assets/
    ├── components/
    │   ├── AudioTrack.tsx
    │   ├── LyricTable.tsx
    │   └── PianoRoll.tsx
    ├── lib/
    │   └── midi.ts
    └── store/
        └── useMidiStore.ts
```
